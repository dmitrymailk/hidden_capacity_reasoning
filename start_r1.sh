model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus '"device=0"' \
    --shm-size 1g \
    -p 1338:80 \
    -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:3.1.1 \
    --model-id $model 