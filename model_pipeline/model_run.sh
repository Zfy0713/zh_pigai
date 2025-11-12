#!/bin/bash

source /mnt/pfs_l2/jieti_team/APP/zhangfengyu/miniconda/bin/activate
conda activate vllm065


cd /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline

# JIUZHANG_CHECKPOINT=/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/models/officials/Qwen2/Qwen2.5-72B-Instruct
JIUZHANG_CHECKPOINT=/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/models/officials/Qwen2/Qwen2.5-7B-Instruct
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

# ImageURL="$1"
ImageURL="https://prod-genie.edstars.com.cn/correct_pipline/processed_image/2025-06-04/0050_18e6e21e-33d2-4927-98bf-eed4548f3393.jpg"
Image_name="${ImageURL##*/}"
path="/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/test_dir/${Image_name}"


LLM_output=$path"/model_output"
if [ -d "$LLM_output" ]; then
    echo $LLM_output
else
    mkdir -p $LLM_output
    echo "The directory '$LLM_output' has been created."
fi

# INPUT=$dir"/ocr_supp_chaiti_rag.csv"
INPUT_PATH="${path}/merge_vl_chaiti/${Image_name}.jsonl"
SAVE_PATH=$LLM_output"/model_output.csv"

# echo "============="
# echo $JIUZHANG_CHECKPOINT
# echo $INPUT
# echo "============="
# CUDA_VISIBLE_DEVICES=1,2,3,4 python -m pigai_pipeline.model_pipeline.model_run \
#     --task_type $TASK_TYPE \
#     --sft_model_path $JIUZHANG_CHECKPOINT \
#     --model_type $MODEL_TYPE \
#     --tokenizer_path $JIUZHANG_CHECKPOINT \
#     --data_path $INPUT \
#     --save_path $SAVE_PATH \
#     --max_gen_length 32768 \
#     --num_return_sequences 1 \
#     --seed 42 \
#     --top_p 0.7 \
#     --top_k 40 \
#     --temperature 0.1 \
#     --repetition_penalty 1.1 \
#     --gpu_memory_utilization 0.7 \
#     --num_gpus 4

# wait


### 线上接口
python -m pigai_pipeline.model_pipeline.qwen-api-2 \
    --input_path $INPUT_PATH \
    --output_path $SAVE_PATH


### 模型后处理 --> pigai图 + draw.csv

MODEL_PATH=$SAVE_PATH
JIAOZHENG_PATH=$path"/jiaozheng/"${Image_name}
PIGAI_PATH=$LLM_output
draw_path=$LLM_output"/draw.csv"

python -m pigai_pipeline.model_pipeline.image_draw_topN_2 \
    --model_output $MODEL_PATH \
    --jiaozheng_path $JIAOZHENG_PATH \
    --pigai_save_path $PIGAI_PATH \
    --draw_path $draw_path