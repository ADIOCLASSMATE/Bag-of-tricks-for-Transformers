# Bag-of-tricks-for-Transformers

# 执行单次训练
``` bash
WANDB_MODE=online \
WANDB_PROJECT=bag-of-tricks-for-transformers \
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
OUTPUT_DIR=./exp/baseline \
uv run torchrun --standalone --nproc_per_node=8 train_gpt.py
```

# 执行完整消融实验
## 代码存放
将修改好的代码存放在：./exp/<trick_name>/train_gpt.py，要求：
- 代码中相对于baseline的修改部分用注释标明，例如：# ablation: add BigramHash
- 其他部分与baseline保持一致，确保消融实验的可比性
- 训练配置（如学习率、batch size等）与baseline保持一致，除非该trick本身涉及训练调度的修改
- 增加./exp/<trick_name>/readme.md，简要说明该trick的出处、实现细节和预期效果

## 实验执行
``` bash
