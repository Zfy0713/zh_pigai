unset http_proxy
unset https_proxy

source /mnt/pfs_l2/jieti_team/APP/zhangfengyu/miniconda/bin/activate
conda activate vllm065

cd /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/search_online

path=/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data2025/1006data/rag

### 9,10库
INPUT_PATH=$path"/merged_ocr_supp_results.csv"
OUTPUT_PATH=$path"/rag_results_online_9_10.jsonl"
SEARCH_NAME="text_vl"
# python bge_online.py \
#     --input_path ${INPUT_PATH} \
#     --save_path ${OUTPUT_PATH} \
#     --top_k 3 --ocr_column $SEARCH_NAME  > $path"/bge.log" 2>&1 &

### 图库
INPUT_PATH=$path"/merged_ocr_supp_results.csv"
OUTPUT_PATH=$path"/rag_results_online_tusou.jsonl"
JIAOZHENG_DIR=$path"/jiaozheng"
SEARCH_NAME="text_vl"

# python tusou_online.py \
#     --input_path ${INPUT_PATH} \
#     --save_path ${OUTPUT_PATH} \
#     --ocr_column $SEARCH_NAME \
#     --jiaozheng_dir ${JIAOZHENG_DIR} > $path"/tusou.log" 2>&1 &

# wait

### rerank
RERANK_MODEL=/mnt/pfs_l2/jieti_team/APP/hegang/models/hegang/models/official/BAAI/bge-reranker-large
BGE_RAG_RESULT=$path"/rag_results_online_9_10.jsonl"
TUSOU_RESULT=$path"/rag_results_online_tusou.jsonl"
ES_RESULT=$path"/es_search.jsonl"

RERANK_SAVE_PATH=$path"/rerank_9_10_tusou_indent4.csv"

python rerank.py \
    --rerank_model_path $RERANK_MODEL \
    --bge_path $BGE_RAG_RESULT \
    --tusou_path $TUSOU_RESULT \
    --top_k 3 \
    --save_path $RERANK_SAVE_PATH > $path"/rerank_9_10_tusou.log" 2>&1 &
    # --es_path $ES_RESULT > 

wait
### 合并

python merge.py \
    --rerank_path $RERANK_SAVE_PATH \
    --ocr_chai_path $path"/merged_ocr_supp_results.csv" \
    --save_path $path"/rag_output.csv"


# nohup bash /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/search_online/rag.sh > /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data2025/0903bei/rag-bge/rerank_9_10_tusou_es.log 2>&1 &