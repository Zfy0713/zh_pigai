### 语文批改pipeline

完整项目路径：
/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline


一键运行
/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/run/run_pipeline.sh

ocr模块：
ocr_pipeline/ocr_pipeline.sh --> merge_vl_chaiti/{img_name}.jsonl （拆题结果）

题库模块
rag_pipeline/rag.sh --> rag_output.jsonl （搜索结果）
输入：INPUT_PATH 拆题结果merge_vl_chaiti路径下

大模型模块
model_pipeline/model_run.sh 大模型批改+后处理 --> model_output/draw.csv （模型批改结果）

后处理模糊匹配（拼音词库批改+线上题库批改）：
mhpp/mhpp_pipeline.sh --> pinyin_ciku/draw.csv + rag/draw.csv （拼音批改 + 题库批改结果）


教辅整页模块：
smartmatch/zhengye.sh --> smart_match/draw.csv （整页比对批改结果）


评测模块：
eval/eval2.sh --> 评测图片 + 指标计算（final_result2.csv/per_pic.csv)

