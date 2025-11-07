#!/bin/bash

source /mnt/pfs_l2/jieti_team/APP/zhangfengyu/miniconda/bin/activate

cd /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline
path=/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/test_dir/fa9c5ca7-573e-4946-993d-7b3cf94f1e28.jpg
Image_name="${path##*/}"


MODEL_PATH=$path"/model_output/model_output.csv"
JIAOZHENG_PATH=$path"/jiaozheng/"${Image_name}
PIGAI_PATH=$path"/model_output"
draw_path=$path"/model_output/draw.csv"

python -m pigai_pipeline.model_pipeline.image_draw_topN_2 \
    --model_output $MODEL_PATH \
    --jiaozheng_path $JIAOZHENG_PATH \
    --pigai_save_path $PIGAI_PATH \
    --draw_path $draw_path


# gt=$path"gt.Json"

# pic_path=$PIGAI_PATH
# pic_eval_path=$path"eval/"


# rand_path=$path"/url_dropDup.txt"
# rand_path=/mnt/pfs/jinfeng_team/APP/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/1127data/1127-300-vl-pigai/url_dropDup.txt
# pic_eval_path_100=$path"eval/"

# python eval2.py $gt $draw_path $pic_path $pic_eval_path_100 $mhpp_path $rand_path
# python eval2.py $gt $draw_path $pic_path $pic_eval_path
# python eval2.py $gt $draw_path $pic_path $pic_eval_path # online test result

### badcase ocr
# python vis_badcase.py $path $rand_path