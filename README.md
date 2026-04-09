# Bag-of-tricks-for-Transformers
``` bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
uv run torchrun --standalone --nproc_per_node=1 train_gpt.py
```