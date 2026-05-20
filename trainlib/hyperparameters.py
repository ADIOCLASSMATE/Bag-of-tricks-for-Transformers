import os
import uuid
from pathlib import Path


# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    # Data paths
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb-edu_100BT_gpt2")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.environ.get("VAL_FILES", os.path.join(data_path, "fineweb_val_*.bin"))
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "gpt2")
    tokenizer_type = os.environ.get("TOKENIZER_TYPE", "gpt2")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    output_dir = os.environ.get("OUTPUT_DIR", str(Path(__file__).parent / "logs"))
    experiment_name = os.environ.get("EXPERIMENT_NAME", run_id)
    control_mode = os.environ.get("CONTROL_MODE", "single_run")
    target_train_tokens = int(os.environ.get("TARGET_TRAIN_TOKENS", "0"))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 50))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", "1"))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model shape
    vocab_size = int(os.environ.get("VOCAB_SIZE", 50257))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "0")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))

    # Optimizer hyperparameters (Muon + Adam, from modded-nanogpt)
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 800))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "0.1"))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Optional W&B logging
    enable_wandb = bool(int(os.environ.get("ENABLE_WANDB", "1")))
    wandb_project = os.environ.get("WANDB_PROJECT", "bag-of-tricks-for-transformers")
    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", run_id)
    wandb_group = os.environ.get("WANDB_GROUP")
    wandb_tags = [tag.strip() for tag in os.environ.get("WANDB_TAGS", "").split(",") if tag.strip()]
    wandb_notes = os.environ.get("WANDB_NOTES")
    wandb_mode = os.environ.get("WANDB_MODE", "offline")
    wandb_dir = os.environ.get("WANDB_DIR")
    wandb_upload_artifacts = bool(int(os.environ.get("WANDB_UPLOAD_ARTIFACTS", "0")))


def hyperparameters_to_dict(args: Hyperparameters) -> dict[str, object]:
    return {
        key: getattr(args, key)
        for key, value in type(args).__dict__.items()
        if not key.startswith("_") and not callable(value)
    }