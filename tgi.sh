volume=$PWD/data
model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

docker run --gpus '"device=0"' --shm-size 1g -p  1337:80 -v $volume:/data  \
    ghcr.io/huggingface/text-generation-inference:3.2.3 \
    --model-id $model
