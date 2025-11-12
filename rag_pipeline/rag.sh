unset http_proxy
unset https_proxy

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

rag_dir=$path"/rag"
if [ -d "$rag_dir" ]; then
    echo $rag_dir
else
    mkdir -p $rag_dir
    echo "The directory '$rag_dir' has been created."
fi

### 9,10库
RAG_INPUT_PATH="${path}/merge_vl_chaiti/${Image_name}.jsonl"
BGE_OUTPUT_PATH=$rag_dir"/bge.jsonl"
SEARCH_NAME="text_vl"
python pigai_pipeline/rag_pipeline/bge_online.py \
    --input_path ${RAG_INPUT_PATH} \
    --save_path ${BGE_OUTPUT_PATH} \
    --top_k 3 --ocr_column $SEARCH_NAME  > $rag_dir"/bge.log" 2>&1 &

### 图库
TUSOU_OUTPUT_PATH=$rag_dir"/tusou.jsonl"
IMAGE_PATH="${path}/jiaozheng/${Image_name}" ### 图片路径
SEARCH_NAME="text_vl"
python pigai_pipeline/rag_pipeline/tusou_online.py \
    --input_path ${RAG_INPUT_PATH} \
    --save_path ${TUSOU_OUTPUT_PATH} \
    --ocr_column $SEARCH_NAME \
    --image_path ${IMAGE_PATH} > $rag_dir"/tusou.log" 2>&1 &

wait

### rerank
RERANK_MODEL=/mnt/pfs_l2/jieti_team/APP/hegang/models/hegang/models/official/BAAI/bge-reranker-large
RERANK_SAVE_PATH=$rag_dir"/rerank_9_10_tusou_indent4.csv"
python /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/rag_pipeline/rerank.py \
    --rerank_model_path $RERANK_MODEL \
    --bge_path $BGE_OUTPUT_PATH \
    --tusou_path $TUSOU_OUTPUT_PATH \
    --top_k 3 \
    --save_path $RERANK_SAVE_PATH > $rag_dir"/rerank_9_10_tusou.log" 2>&1 &
wait
python /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/rag_pipeline/merge.py \
    --rerank_path $RERANK_SAVE_PATH \
    --ocr_chai_path $RAG_INPUT_PATH \
    --save_path $rag_dir"/rag_output.csv"


# nohup bash /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/search_online/rag.sh > /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data2025/0903bei/rag-bge/rerank_9_10_tusou_es.log 2>&1 &