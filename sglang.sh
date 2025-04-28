volume=$PWD/data
model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# docker run -d -ti --rm --gpus all \
docker run -ti --rm --gpus all \
    --shm-size 1g \
    -p 1337:30000 \
    -v $volume:/root/.cache/huggingface \
    -v ./:/code \
    -w /code \
    --ipc=host \
    lmsysorg/sglang:v0.4.5-cu125 \
    python3 -m sglang.launch_server --model-path $model --host 0.0.0.0 --port 30000 --attention-backend triton --sampling-backend pytorch --disable-cuda-graph --disable-cuda-graph-padding --disable-custom-all-reduce --disable-mla --random-seed 42 --max-running-request 1
    # python3 -m sglang.launch_server --model-path $model --host 0.0.0.0 --port 30000 
    # --max-running-request 1
    # python3 -m sglang.launch_server --model-path $model --host 0.0.0.0 --port 30000 --dp-size 4