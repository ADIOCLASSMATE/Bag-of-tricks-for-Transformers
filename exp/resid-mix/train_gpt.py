"""
Trick: resid-mix — learnable per-channel residual mixing with the original embedding.

Each Block takes an additional x0 (token embedding output) and mixes it with the
current hidden state before sublayers:
  mix = resid_mix  # shape [2, dim], init [1, 0]
  x = mix[0] * x + mix[1] * x0
This allows each layer to blend the current representation with the original input
embedding. Initialized as [1, 0], so at step 0 behavior is identical to baseline
(pure single-variable architectural ablation).

The resid_mix tensor is 2D so it would otherwise be routed to Muon by the default
optimizer split; we add it to CONTROL_TENSOR_NAME_PATTERNS so it goes to AdamW
(scalar group) instead — Muon is for spectral updates on weight matrices, not for
per-channel scaling vectors.

Architecture: same as baseline (GQA, RoPE, RMSNorm, QK-Norm, untied embeddings,
GELU MLP, standard residual) plus the resid_mix mixing.

Training recipe (identical to baseline):
  - Muon optimizer for 2D matrix params in transformer blocks
  - AdamW for scalar/control params (incl. resid_mix), token embedding, and lm_head
  - Muon momentum warmup
  - Weight decay: 0.0 on token embedding, Muon, and scalar params; 0.1 on lm_head
  - JIT warmup with state reset
"""

from __future__ import annotations

import copy
import glob
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import tiktoken
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# resid-mix uses non-empty CONTROL_TENSOR_NAME_PATTERNS default to keep
# resid_mix params (1D/scalar-like) out of Muon and in Adam.
os.environ.setdefault("CONTROL_TENSOR_NAME_PATTERNS", "resid_mix,resid_mixes")

from trainlib.hyperparameters import Hyperparameters, hyperparameters_to_dict
from trainlib.optimizer import CONTROL_TENSOR_NAME_PATTERNS, Muon
from trainlib.validation import build_sentencepiece_luts, build_gpt2_luts, load_validation_tokens, eval_val
from trainlib.dataloader import DistributedTokenLoader
from trainlib.transformer import (
    RMSNorm, CastedLinear, Rotary, apply_rotary_emb,
    CausalSelfAttention, MLP, restore_low_dim_params_to_fp32,
)


# -----------------------------
# TRANSFORMER MODULES (custom Block + GPT for resid_mix)
# -----------------------------

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base)
        self.mlp = MLP(dim, mlp_mult)
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())  # trick: resid-mix

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:  # trick: resid-mix — takes x0
        mix = self.resid_mix.to(dtype=x.dtype)  # trick: resid-mix
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0  # trick: resid-mix
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        rope_base: float,
    ):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith('.proj'):
                    # Output projections (attn.proj, mlp.proj): zero init
                    nn.init.zeros_(module.weight)
                elif name == 'lm_head':
                    # LM head: small normal
                    nn.init.normal_(module.weight, mean=0.0, std=0.001)
                elif name.endswith(('.c_q', '.c_k', '.c_v', '.fc', '.gate_proj', '.up_proj', '.attn_gate', '.key_proj')):
                    # Q/K/V and MLP fc: PyTorch default, skip
                    pass
                else:
                    # Other Linear layers: Normal(0, 0.02)
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x0 = x  # trick: resid-mix — save embedding for mixing
        for block in self.blocks:
            x = block(x, x0)  # trick: resid-mix
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits = self.lm_head(x)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    # Monkey-patch the function in trainlib.optimizer so Muon (defined there)
    # sees the compiled version — a plain "global" only rebinds the import
    # alias in this module, not the one Muon actually calls.
    import trainlib.optimizer
    trainlib.optimizer.zeropower_via_newtonschulz5 = torch.compile(
        trainlib.optimizer.zeropower_via_newtonschulz5
    )

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    grad_accum_steps = args.grad_accum_steps
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs(os.path.join(args.output_dir, "log"), exist_ok=True)
        logfile = os.path.join(args.output_dir, "log", f"{args.run_id}.txt")
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    wandb = None
    wandb_run = None

    def wandb_log(metrics: dict[str, object], *, step: int | None = None) -> None:
        if wandb_run is None:
            return
        if step is None:
            wandb_run.log(metrics)
        else:
            wandb_run.log(metrics, step=step)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.tokenizer_type == "gpt2":
        enc = tiktoken.get_encoding("gpt2")
        if enc.n_vocab != args.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={args.vocab_size} does not match GPT-2 vocab_size={enc.n_vocab}"
            )
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_gpt2_luts(
            enc, args.vocab_size, device
        )
    elif args.tokenizer_path.endswith(".model"):
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        if int(sp.vocab_size()) != args.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
            )
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
            sp, args.vocab_size, device
        )
    else:
        raise ValueError(
            f"Unknown tokenizer: tokenizer_type={args.tokenizer_type}, "
            f"tokenizer_path={args.tokenizer_path}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    log0(f"val_bpb:enabled tokenizer_kind={args.tokenizer_type} tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        rope_base=args.rope_base,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr, "weight_decay": 0.0}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=0.0,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    optimizer_head: torch.optim.Optimizer | None = None
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.AdamW(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr, "weight_decay": args.weight_decay}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    if master_process and args.enable_wandb and args.wandb_mode != "disabled":
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ENABLE_WANDB=1 but wandb is not installed. Install dependencies or set ENABLE_WANDB=0."
            ) from exc
        wandb_init_kwargs: dict[str, object] = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "mode": args.wandb_mode,
            "config": {
                **hyperparameters_to_dict(args),
                "script_name": Path(__file__).name,
                "dataset_dir_name": dataset_dir.name,
                "actual_train_files": actual_train_files,
                "world_size": world_size,
                "grad_accum_steps": grad_accum_steps,
                "device_type": device.type,
                "torch_version": torch.__version__,
                "python_version": sys.version.split()[0],
                "model_params": int(n_params),
            },
            "tags": args.wandb_tags,
            "reinit": False,
        }
        if args.wandb_entity:
            wandb_init_kwargs["entity"] = args.wandb_entity
        if args.wandb_group:
            wandb_init_kwargs["group"] = args.wandb_group
        if args.wandb_notes:
            wandb_init_kwargs["notes"] = args.wandb_notes
        if args.wandb_dir:
            wandb_init_kwargs["dir"] = args.wandb_dir
        wandb_run = wandb.init(**wandb_init_kwargs)
        wandb_log(
            {
                "setup/model_params": int(n_params),
                "setup/world_size": world_size,
                "setup/grad_accum_steps": grad_accum_steps,
                "setup/train_shards": actual_train_files,
                "setup/val_tokens": int(val_tokens.numel() - 1),
            },
            step=0,
        )

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            wandb_log(
                {
                    "val/loss": float(val_loss),
                    "val/bpb": float(val_bpb),
                    "train/time_ms": float(training_time_ms),
                    "train/step_avg_ms": float(training_time_ms / max(step, 1)),
                },
                step=step,
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
            wandb_log(
                {
                    "train/loss": float(train_loss.item()),
                    "train/time_ms": float(approx_training_time_ms),
                    "train/step_avg_ms": float(approx_training_time_ms / step),
                    "train/lr_scale": float(scale),
                    "train/muon_momentum": float(muon_momentum),
                    "train/lr_tok": float(optimizer_tok.param_groups[0]["lr"]),
                    "train/lr_head": float(optimizer_head.param_groups[0]["lr"]) if optimizer_head is not None else 0.0,
                    "train/lr_matrix": float(optimizer_muon.param_groups[0]["lr"]),
                    "train/lr_scalar": float(optimizer_scalar.param_groups[0]["lr"]),
                    "train/tokens_seen": int(step * args.train_batch_tokens),
                },
                step=step,
            )

        # Check wallclock cap
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    peak_mem_allocated_mib = torch.cuda.max_memory_allocated() // 1024 // 1024
    peak_mem_reserved_mib = torch.cuda.max_memory_reserved() // 1024 // 1024
    log0(
        f"peak memory allocated: {peak_mem_allocated_mib} MiB "
        f"reserved: {peak_mem_reserved_mib} MiB"
    )
    wandb_log(
        {
            "system/peak_mem_allocated_mib": int(peak_mem_allocated_mib),
            "system/peak_mem_reserved_mib": int(peak_mem_reserved_mib),
            "final/stopped_early": bool(stop_after_step is not None and step < args.iterations),
            "final/step": int(step),
        },
        step=step,
    )

    # -----------------------------
    # SERIALIZATION + FINAL EVALUATION
    # -----------------------------

    if master_process:
        model_path = os.path.join(args.output_dir, "final_model.pt")
        torch.save(base_model.state_dict(), model_path)
        model_bytes = os.path.getsize(model_path)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")
        wandb_log(
            {
                "artifact/model_bytes": int(model_bytes),
                "artifact/code_bytes": int(code_bytes),
                "artifact/submission_bytes_raw": int(model_bytes + code_bytes),
            },
            step=step,
        )
    else:
        model_path = os.path.join(args.output_dir, "final_model.pt")

    if distributed:
        dist.barrier()
    torch.cuda.synchronize()
    t_feval = time.perf_counter()
    final_val_loss, final_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    final_eval_time_ms = 1000.0 * (time.perf_counter() - t_feval)
    log0(
        f"final val_loss:{final_val_loss:.4f} val_bpb:{final_val_bpb:.4f} "
        f"eval_time:{final_eval_time_ms:.0f}ms"
    )
    log0(f"final_exact val_loss:{final_val_loss:.8f} val_bpb:{final_val_bpb:.8f}")
    wandb_log(
        {
            "final/val_loss": float(final_val_loss),
            "final/val_bpb": float(final_val_bpb),
            "final/eval_time_ms": float(final_eval_time_ms),
        },
        step=step,
    )
    if wandb_run is not None:
        wandb_run.summary["val_loss"] = float(final_val_loss)
        wandb_run.summary["val_bpb"] = float(final_val_bpb)
        wandb_run.summary["train_steps"] = int(step)
        wandb_run.summary["peak_mem_allocated_mib"] = int(peak_mem_allocated_mib)
        wandb_run.summary["peak_mem_reserved_mib"] = int(peak_mem_reserved_mib)
        wandb_run.finish()

    if master_process:
        actual_train_tokens = int(step * args.train_batch_tokens)
        stopped_early = stop_after_step is not None and step < args.iterations
        metrics_payload = {
            "version": 1,
            "run_id": args.run_id,
            "trainer": Path(__file__).name,
            "experiment_name": args.experiment_name,
            "hyperparameters": hyperparameters_to_dict(args),
            "control": {
                "mode": args.control_mode,
                "target_train_tokens": args.target_train_tokens,
                "actual_train_tokens": actual_train_tokens,
                "target_wallclock_seconds": args.max_wallclock_seconds if args.control_mode == "fixed_compute" else 0.0,
                "actual_wallclock_seconds": training_time_ms / 1000.0,
                "stopped_early": stopped_early,
            },
            "model": {
                "num_layers": args.num_layers,
                "model_dim": args.model_dim,
                "num_heads": args.num_heads,
                "num_kv_heads": args.num_kv_heads,
                "mlp_mult": args.mlp_mult,
                "vocab_size": args.vocab_size,
                "model_params": int(n_params),
            },
            "training": {
                "iterations": args.iterations,
                "train_batch_tokens": args.train_batch_tokens,
                "train_seq_len": args.train_seq_len,
                "world_size": world_size,
                "grad_accum_steps": grad_accum_steps,
                "max_wallclock_seconds": args.max_wallclock_seconds,
                "final_step": int(step),
                "training_time_ms": float(training_time_ms),
            },
            "metrics": {
                "final_val_loss": float(final_val_loss),
                "final_val_bpb": float(final_val_bpb),
                "final_eval_time_ms": float(final_eval_time_ms),
                "peak_mem_allocated_mib": int(peak_mem_allocated_mib),
                "peak_mem_reserved_mib": int(peak_mem_reserved_mib),
            },
            "data": {
                "data_path": args.data_path,
                "tokenizer_path": args.tokenizer_path,
                "train_shards": actual_train_files,
            },
            "artifacts": {
                "model_path": str(model_path),
                "log_path": logfile,
            },
        }
        metrics_path = Path(args.output_dir) / "result.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
        log0(f"Wrote final metrics to {metrics_path}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
