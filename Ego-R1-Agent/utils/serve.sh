export CUDA_VISIBLE_DEVICES=0,1

# vllm serve Qwen/Qwen2.5-3B-Instruct --port=23332 \
#     --tensor-parallel-size=2 \
#     --gpu-memory-utilization=0.7 \
#     --disable-custom-all-reduce


vllm serve Ego-R1/Ego-R1-Agent-3B --port=23333 \
    --tensor-parallel-size=2 \
    --gpu-memory-utilization=0.7 \
    --disable-custom-all-reduce