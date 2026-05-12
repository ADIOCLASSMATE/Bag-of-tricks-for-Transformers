# Experiment Errata — Bag of Tricks for Transformers

2026-05-12

## 1. Adam → AdamW + Weight Decay 策略修正

### 发现的问题

所有 46 个实验的 `train_gpt.py` 均使用 `torch.optim.Adam`（非 `AdamW`），且完全未设置 `weight_decay`。baseline 代码明确注释 "No weight decay, no grad clip"。

### 为什么需要改

- `Adam` 的 weight decay 是 L2 正则化（耦合在梯度中），`AdamW` 是解耦 weight decay。现代训练统一使用 `AdamW`。
- 无 weight decay 在小模型上可能不明显，但 scale 到 medium/large 时 embedding 和 output head 容易过拟合。

### 修改内容

| 参数组 | 优化器 | weight_decay | 原因 |
|--------|--------|:---:|------|
| `optimizer_tok` (untied 时) | AdamW | **0** | 仅做 input lookup，无需正则化 |
| `optimizer_tok` (tied 时) | AdamW | **0.1** | embedding 同时是 output projection，需要正则化 |
| `optimizer_head` (untied 时) | AdamW | **0.1** | output projection，需要正则化 |
| `optimizer_scalar` | AdamW | **0** (显式) | bias/norm/1D 参数不加 weight decay（标准做法） |
| `optimizer_muon` | Muon | N/A | Muon 的 Newton-Schulz 正交化自带正则化 |
| `optimizer_engram` | AdamW | **0** (显式) | embedding 类参数，仅做 lookup |
| `optimizer_conv` | AdamW | **0** (显式) | Conv1d 权重，非 output projection |

### ⚠️ AdamW 默认值陷阱

`torch.optim.AdamW` 的默认 `weight_decay=0.01`（非 0）。初次迁移时未显式传 `weight_decay=0.0`，导致 `optimizer_scalar` 等无意中获得了 0.01 的 weight decay。**任何从 Adam → AdamW 的迁移都必须显式设置 weight_decay。**

---

## 2. Embedding 初始化修正（N(0,1) → N(0, 0.02)）

### 发现的问题

`nn.Embedding` 的 PyTorch 默认初始化是 N(0, 1)。当 `tie_embeddings=True` 时，embedding 矩阵同时充当 output projection，但初始化未按 output projection 的标准（Xavier, 1/√d）来。

**后果**：logit std ≈ √512 × 1.0 ≈ 22.6 → softmax 极度尖锐 → step 0 val_loss ≈ 429（理论最小值 ln(1024) ≈ 6.93）。

### 为什么 GPT-2/nanoGPT/Qwen3 没有这个问题

这些项目全部使用 `_init_weights` 显式设置 `nn.init.normal_(weight, std=0.02)`（GPT-2 用 0.02，HuggingFace 用 `initializer_range=0.02`）。你的 baseline 恰好跳过了这一步，使用了 PyTorch 的出厂默认值 N(0,1)。

### 修改内容

在所有 46 个实验的 `GPT` 类中添加 HuggingFace 风格的初始化：

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

# 在 GPT.__init__ 末尾调用
self.apply(self._init_weights)
```

- `CastedLinear(nn.Linear)` 自动被 `isinstance(module, nn.Linear)` 匹配
- `RMSNorm`: 该代码库的 RMSNorm 无任何可学习参数（仅 `self.eps`），`_init_weights` 自然不会触达
- `nn.Conv1d` (engram) 不被匹配，保持显式 zero-init
- `nn.Parameter` (lambdas 等) 不被 `self.apply()` 遍历

### 特殊实验处理

5 个已有自定义 `_init_weights` 的实验（`small-embed-init`, `ortho-init`, `zero-init`, `parameter-golf-baseline`, `mup-scaled-output-projections`）改为"baseline init + trick override"模式。

### 效果验证

| | 修复前 | 修复后 |
|---|:---:|:---:|
| step 0 val_loss (tied) | 429.5 | **7.03** |
| step 0 val_loss (untied) | 7.1 | **7.05** |

---

## 3. 默认 Tied → Untied Embeddings

### 发现的问题

Tied vs Untied 消融实验（10B tokens，N(0,0.02) init）结果：

| 模型规模 | Tied bpb | Untied bpb | Δ |
|------|:---:|:---:|:---:|
| Small (9L/512d, 17M) | 1.256 | 1.245 | -0.011 |
| Medium (18L/1024d, 133M) | 1.105 | 1.079 | -0.025 |

Untied 在所有规模上均优于 Tied，且差距随模型增大而扩大。

### 修改内容

全部 46 个实验的 `TIE_EMBEDDINGS` 默认值从 `"1"` 改为 `"0"`：

- **44 个实验** → untied 默认
- **`small-embed-init`** → 保持 `"1"`（trick 仅在 tied 时激活）
- **`untie-embed`** → 改为 `"1"`（作为 tied 对照组）

代码本身已完全 conditional（`self.lm_head` 创建、`optimizer_head` 创建、`weight_decay` 分配、LR 选择），无需其他逻辑修改。

### 额外效果

切换到 untied 后：
- 输入 embedding 独立于输出 head → 无 logit 方差爆炸风险
- `weight_decay` 自然地仅作用于 output head（lm_head），输入 embedding 不受影响
- 与现代 LLM 实践（LLaMA, Mistral, Qwen2, Gemma 等）一致

---

## 4. 其他修复

### `grad_accum_steps` 硬编码

`grad_accum_steps = 1` 硬编码在 `main()` 中，改为从 `args.grad_accum_steps` 读取，并在 `Hyperparameters` 类中添加对应环境变量。

### `def forward` 缩进损坏

批量修改脚本在处理 `_init_weights` 插入时误将 `GPT.forward` 的缩进去掉（从类方法变成模块级函数），导致 `torch.compile` 报 `NotImplementedError`。已批量修复。

### multi-ve `torch.compile` 兼容性

`multi-ve` 实验在 `torch.compile` 阶段报 `aten.view.default` shape 错误。此 bug 为已有问题（git stash 还原后同样报错），与本次修改无关。

---

## 修改影响范围

| 文件 | 修改内容 |
|------|---------|
| 46 个 `exp/*/train_gpt.py` | Adam→AdamW, weight_decay 策略, N(0,0.02) init, TIE_EMBEDDINGS 默认→0 |
| `exp/run_experiments.py` | ENV_KEY_MAP 添加 `weight_decay` |
| `exp/baseline/baseline-tie-vs-untie.json` | 新增 tied-vs-untied 消融 manifest |

## Wandb 消融结果

- Entity: `Bag-of-Tricks`
- Project: `bag-of-tricks-for-transformers`
- Group: `tie-vs-untie`
- 4 runs: tied-small-10B, untied-small-10B, tied-medium-10B, untied-medium-10B
