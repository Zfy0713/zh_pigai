#!/bin/bash

source /mnt/pfs_l2/jieti_team/APP/zhangfengyu/miniconda/bin/activate
conda activate vllm065
# conda activate vllm010
# conda activate qwen3
# conda activate qwen3_chat

JIUZHANG_CHECKPOINT=/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/models/officials/Qwen2/Qwen2.5-72B-Instruct
# JIUZHANG_CHECKPOINT=/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/models/officials/ByteDance-Seed/Seed-OSS-36B-Instruct
# JIUZHANG_CHECKPOINT=/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/models/officials/Qwen3/Qwen3-32B
# JIUZHANG_CHECKPOINT=/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/models/officials/Qwen3/Qwen3-30B-A3B

# TASK_TYPE=pigai_yuwen_json_input ### 输入json格式的拆录结果
TASK_TYPE=pigai_yuwen
# TASK_TYPE=jieti_yuwen ### 先解题后批改
# TASK_TYPE=pigai_yuwen_new ### 0409更新prompt
# TASK_TYPE=pigai_yuwen_vl_ocr ### 7b-ocr加手写体
# 支持qwen
MODEL_TYPE=qwen

dir=/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/test_dir/merge_vl_chaiti
if [ -d "$dir" ]; then
    echo $dir
else
    mkdir -p $dir
    echo "The directory '$dir' has been created."
fi

# INPUT=$dir"/ocr_supp_chaiti_rag.csv"
INPUT=$dir"/85bf7b88-141d-430c-98d0-75ba641c26cd.jpg.jsonl"
SAVE_PATH=$dir"/model_output.csv"

echo "============="
echo $JIUZHANG_CHECKPOINT
echo $INPUT
echo "============="
CUDA_VISIBLE_DEVICES=1,2,3,4 python \
    /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/model_pipeline/model_run.py \
    --task_type $TASK_TYPE \
    --sft_model_path $JIUZHANG_CHECKPOINT \
    --model_type $MODEL_TYPE \
    --tokenizer_path $JIUZHANG_CHECKPOINT \
    --data_path $INPUT \
    --save_path $SAVE_PATH \
    --max_gen_length 32768 \
    --num_return_sequences 1 \
    --seed 42 \
    --top_p 0.7 \
    --top_k 40 \
    --temperature 0.1 \
    --repetition_penalty 1.1 \
    --gpu_memory_utilization 0.9 \
    --num_gpus 4

