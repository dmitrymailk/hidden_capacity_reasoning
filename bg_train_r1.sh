export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM="true"

python -m train_r1_compression > hidden_capacity_reasoning.log 2>&1 &