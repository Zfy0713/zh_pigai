#!/bin/bash

source /mnt/pfs_l2/jieti_team/APP/zhangfengyu/miniconda/bin/activate
conda activate vllm065

cd /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline


# ImageURL="https://prod-genie.edstars.com.cn/correct_pipline/processed_image/2025-06-04/0050_18e6e21e-33d2-4927-98bf-eed4548f3393.jpg"
# Image_name="${ImageURL##*/}"
# DIR="/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/test_dir/${Image_name}"

ImageURL="$1"
PIGAI_DIR="$2"
Image_name="${ImageURL##*/}"
DIR="${PIGAI_DIR}/${Image_name}"

PIGAI_DIR="${DIR}/pigai"
if [ -d "$PIGAI_DIR" ]; then
    echo $PIGAI_DIR
else
    mkdir -p $PIGAI_DIR
    echo "The directory '$PIGAI_DIR' has been created."
fi

### 批改渲染
python -m pigai_pipeline.eval.image_draw \
    --pigai_dir $DIR \
    --save_img_path "${PIGAI_DIR}/${Image_name}" \
    --save_merged_draw_path "${PIGAI_DIR}/merged_draw.csv"


### 自动化评测
EVAL_DIR="${PIGAI_DIR}/eval/"
if [ -d "$EVAL_DIR" ]; then
    echo $EVAL_DIR
else
    mkdir -p $EVAL_DIR
    echo "The directory '$EVAL_DIR' has been created."
fi

# gt=$PIGAI_DIR"/gt.Json"
gt="$3"
draw_path="${PIGAI_DIR}/merged_draw.csv"
pic_path=$PIGAI_DIR"/" ### 批改效果集合
pic_eval_path=$EVAL_DIR ### 评测效果集合

python -m pigai_pipeline.eval.eval2 $gt $draw_path $pic_path $pic_eval_path