#!/bin/bash

source /mnt/pfs_l2/jieti_team/APP/zhangfengyu/miniconda/bin/activate
conda activate vllm065

cd /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline

path=/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/test_dir/fa9c5ca7-573e-4946-993d-7b3cf94f1e28.jpg


### 拼音词库批改
CIKU_PATH=/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v2/pinyin_large/pinyin_dict.json
PINYIN_OCR=$path"/pinyin_ciku/pinyin_output.jsonl"
SAVE_PATH=$path"/pinyin_ciku/draw.jsonl"

python -m pigai_pipeline.mhpp.pinyin.pinyin_run \
    --ciku_path $CIKU_PATH \
    --pinyin_supp_ocr $PINYIN_OCR \
    --save_path $SAVE_PATH


### 题库批改
INPUT_PATH=$path"/rag/rag_output.jsonl"
SAVE_PATH=$path"/rag/draw.jsonl"

python -m pigai_pipeline.mhpp.rag_mhpp.run \
    --input_path $INPUT_PATH \
    --save_path $SAVE_PATH

### 整页教辅批改