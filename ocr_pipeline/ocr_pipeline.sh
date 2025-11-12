#!/bin/bash

source /mnt/pfs_l2/jieti_team/APP/zhangfengyu/miniconda/bin/activate
conda activate vllm065

cd /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline

# ImageURL="https://ss-prod-genie.oss-cn-beijing.aliyuncs.com/correct_pipeline/processed_image/2025-09-16/fa9c5ca7-573e-4946-993d-7b3cf94f1e28.jpg"
ImageURL="$1"
PIGAI_DIR="$2"
Image_name="${ImageURL##*/}"
save_path="${PIGAI_DIR}/${Image_name}"

python ocr_pipeline/vl_ocr.py \
    --image_url $ImageURL \
    --path $save_path 