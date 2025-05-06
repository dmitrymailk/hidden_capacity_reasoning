export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM="true"

python -m train_r1_compression_full > hidden_capacity_reasoning_full.log 2>&1 &