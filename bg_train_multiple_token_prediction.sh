export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM="true"

python -m train_multiple_token_prediction > train_multiple_token_prediction.log 2>&1 &