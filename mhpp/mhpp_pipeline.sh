#!/bin/bash

source /mnt/pfs_l2/jieti_team/APP/zhangfengyu/miniconda/bin/activate
conda activate vllm065

cd /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline

# ImageURL="$1"
# ImageURL="https://prod-genie.edstars.com.cn/correct_pipline/processed_image/2025-06-04/0050_18e6e21e-33d2-4927-98bf-eed4548f3393.jpg"
# Image_name="${ImageURL##*/}"
# path="/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/test_dir/${Image_name}"

ImageURL="$1"
PIGAI_DIR="$2"
Image_name="${ImageURL##*/}"
path="${PIGAI_DIR}/${Image_name}"

### 拼音词库批改
CIKU_PATH=/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v2/pinyin_large/pinyin_dict.json
PINYIN_OCR=$path"/pinyin_ciku/pinyin_output.jsonl"
PINYIN_PIGAI_OUTPUT=$path"/pinyin_ciku/draw.jsonl"

python -m pigai_pipeline.mhpp.pinyin.pinyin_run \
    --ciku_path $CIKU_PATH \
    --pinyin_supp_ocr $PINYIN_OCR \
    --save_path $PINYIN_PIGAI_OUTPUT


### 题库批改
INPUT_PATH=$path"/rag/rag_output.jsonl"
SAVE_PATH=$path"/rag/draw.jsonl"

python -m pigai_pipeline.mhpp.rag_mhpp.run \
    --input_path $INPUT_PATH \
    --save_path $SAVE_PATH