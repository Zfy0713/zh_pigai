#!/bin/bash

source /mnt/pfs_l2/jieti_team/APP/zhangfengyu/miniconda/bin/activate
conda activate vllm065

cd /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline

# ImageURL="https://ss-prod-genie.oss-cn-beijing.aliyuncs.com/correct_pipeline/processed_image/2025-09-16/fa9c5ca7-573e-4946-993d-7b3cf94f1e28.jpg"
ImageURL="https://prod-genie.edstars.com.cn/correct_pipline/processed_image/2025-06-04/3f58_bcad8847-b54c-4c29-9cfa-bcab361e2297.jpg"
Image_name="${ImageURL##*/}"
save_path="test_dir/"${Image_name}

python ocr_pipeline/vl_ocr.py \
    --image_url $ImageURL \
    --path $save_path 