
ImageURL="https://ss-prod-genie.oss-cn-beijing.aliyuncs.com/correct_pipeline/processed_image/2025-09-16/58910c4a-67e7-4ed4-bd96-42593b30753b.jpg"
pigai_dir="/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/test_dir"

### ocr
# bash /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/ocr_pipeline/ocr_pipeline.sh $ImageURL $pigai_dir

### llm
# bash /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/model_pipeline/model_run.sh $ImageURL $pigai_dir

### rag
# bash /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/rag_pipeline/rag.sh $ImageURL $pigai_dir

### 后处理(词库+题库)
# bash /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/mhpp/mhpp_pipeline.sh $ImageURL $pigai_dir

### 整页批改
# bash /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/smartmatch/zhengye.sh $ImageURL $pigai_dir

### 评测（确保gt文件在【${pigai_dir}/pigai】目录下）
gt="/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data2025/0925jiaofu/merge_vlocr/gt.Json"
bash /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/eval/eval2.sh $ImageURL $pigai_dir $gt