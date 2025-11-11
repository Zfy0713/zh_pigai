source /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/.bashrc

cd /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline


# ImageURL="$1"
ImageURL=https://ss-prod-genie.oss-cn-beijing.aliyuncs.com/correct_pipeline/processed_image/2025-09-16/a28fb325-488e-4af7-a071-dcb56274ba8c.jpg
Image_name="${ImageURL##*/}"
path="/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/test_dir/${Image_name}"

### 整页搜索
IMAGE_PATH="${path}/jiaozheng/${Image_name}"
SMART_DIR="${path}/smart_match"
if [ -d "$SMART_DIR" ]; then
    echo $SMART_DIR
else
    mkdir -p $SMART_DIR
    echo "The directory '$SMART_DIR' has been created."
fi
SREACH_PATH="${SMART_DIR}/smart_match_res.json"
COMPARE_IMG_PATH="${SMART_DIR}/compare_${Image_name}"
python pigai_pipeline/smartmatch/search.py \
    --img_path $IMAGE_PATH \
    --search_save_path $SREACH_PATH \
    --compare_img_path $COMPARE_IMG_PATH


### 处理整页结果
PROCESSED_PATH="${SMART_DIR}/smart_match_processed.json"
python pigai_pipeline/smartmatch/SmartMatch_Processor.py \
    --search_json_path $SREACH_PATH \
    --processed_save_path $PROCESSED_PATH


### 题目搜索（当前页内）
CHAITI_PATH="${path}/merge_vl_chaiti/${Image_name}.jsonl"
RAG_OUTPUT="${SMART_DIR}/rag_output.jsonl"

python -m pigai_pipeline.smartmatch.Query_match \
    --input_chaiti_file $CHAITI_PATH \
    --input_search_file $PROCESSED_PATH \
    --output_file $RAG_OUTPUT

### 答案比对
DRAW_PATH="${SMART_DIR}/draw.jsonl"
python -m pigai_pipeline.smartmatch.pigai_mhpp \
    --input_file $RAG_OUTPUT \
    --save_path $DRAW_PATH