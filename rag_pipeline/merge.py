import csv
import sys
import os
import json
import pandas as pd
import ast
from file_utils import load_jsonL

def load_csv_2_dict(csv_path):
    # 增加字段大小限制
    # csv.field_size_limit(sys.maxsize)
    data = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        col_names = next(reader)
        csv_reader = csv.DictReader(f, col_names)
        for row in csv_reader:
            d = {}
            for k,v in row.items():
                d[k]=v
            data.append(d)
    return data


def main(args):
    rag_csv_path = args.rerank_path ### 搜题结果
    chaiti_path = args.ocr_chai_path ### 拆题结果
    save_path = args.save_path

    # data = load_csv_2_dict(chaiti_csv_path)
    data = load_jsonL(chaiti_path)
    rag_data = load_csv_2_dict(rag_csv_path)
    assert len(data) == len(rag_data)
    outputs = []

    for a, r in zip(rag_data, data):
        # idx = r["序号"]
        # url = r['img_url'] if 'img_url' in r else r['url']
        # vertices = r["vertices"]
        # title = r["title"]
        # ocr = r["ocr"]
        # hand_text_list = r["hand_text_list"]

        
        try:

            combined_rag = eval(a["topk"])
            # combined_rag = ast.literal_eval(a["topk"])
        except Exception as e:
            print(f"Error evaluating topk for index {a}: {e}")
            combined_rag = []
        
        # current_line = {
        #     "序号": idx,
        #     "url": url,
        #     "vertices": vertices,
        #     "title": title,
        #     "ocr": ocr,
        #     "hand_text_list": hand_text_list,
        #     "combined_rag": combined_rag
        # }
        r.update({"combined_rag": combined_rag})
        outputs.append(r)

    if save_path.endswith('.csv'):
        df = pd.DataFrame(outputs)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')

        json_path = save_path.replace('.csv', '.jsonl')

        with open(json_path, 'w') as f:
            for d in outputs:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

        print(f"Saved merged results to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank_path", type=str, required=True, help="rerank结果文件路径")
    parser.add_argument("--ocr_chai_path", type=str, required=True, help="ocr拆题结果文件路径")
    parser.add_argument("--save_path", type=str, required=True, help="合并后保存路径")
    args = parser.parse_args()
    main(args)