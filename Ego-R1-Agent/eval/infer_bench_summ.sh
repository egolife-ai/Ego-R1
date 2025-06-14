#!/bin/bash

export VIDEO_LLM_URL=http://127.0.0.1:7060/video_llm
export VLM_URL=http://127.0.0.1:7090/vlm

datetime=$(date +%Y%m%d_%H%M%S)
max_turns=7
# Create log directory if it doesn't exist
mkdir -p infer_logs

model_name_or_path=sieufgsb9dv77w-94r/qwen25-3b-it-sft4500-len8192-rl-bs4-gs145
# sieufgsb9dv77w-94r/qwen25-3b-it-sft4500-len8192-rl-bs4-gs145
# 184 have some issues
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python eval/infer_bench_summ.py \
    --bench_name video-mme-long \
    --max_turns ${max_turns} \
    --data_start 620 \
    --data_end -1 \
    --model_name_or_path ${model_name_or_path} \
    --dataset ./benchmarks/video-mme-long.parquet \
    | tee infer_logs/log_video-mme-long_summ_mt${max_turns}-${datetime}_s620-e-1.log 2>&1

# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python eval/infer_bench_summ.py \
#     --bench_name video-mme-long \
#     --max_turns ${max_turns} \
#     --data_start 765 \
#     --data_end 810 \
#     --dataset ./benchmarks/video-mme-long.parquet \
#     | tee infer_logs/log_video-mme-long_summ_mt${max_turns}-${datetime}_s765-e810.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python eval/infer_bench_summ.py \
#     --bench_name video-mme-long \
#     --max_turns ${max_turns} \
#     --data_start 810 \
#     --data_end 855 \
#     --dataset ./benchmarks/video-mme-long.parquet \
#     | tee infer_logs/log_video-mme-long_summ_mt${max_turns}-${datetime}_s810-e855.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python eval/infer_bench_summ.py \
#     --bench_name video-mme-long \
#     --max_turns ${max_turns} \
#     --data_start 855 \
#     --data_end 900 \
#     --dataset ./benchmarks/video-mme-long.parquet \
#     | tee infer_logs/log_video-mme-long_summ_mt${max_turns}-${datetime}_s855-e900.log 2>&1 &
