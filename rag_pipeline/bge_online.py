from utils.util.send_sign_http import send_request
import time
import requests
import csv
import copy
import base64
import json
import sys
import os
import re

def load_csv_2_dict(csv_path):
    # 增加字段大小限制
    csv.field_size_limit(sys.maxsize)
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        col_names = next(reader)
        csv_reader = csv.DictReader(f, col_names)
        for row in csv_reader:
            d = {}
            for k,v in row.items():
                d[k]=v
            data.append(d)
    return data

def my_request(http_url,url_params,body_params,method, stream = False):
    header = 'application/json'
    # online config
    # ACCESS_KEY_ID = "1300790167388291072"
    # ACCESS_KEY_SECRET = "a0219c47e2f149708c9caf261b3b7367"
    ACCESS_KEY_ID = "1384834886006738944"
    ACCESS_KEY_SECRET = "3d213a6b4a56487191eb1a47f63b096b"

    access_key_id = ACCESS_KEY_ID
    access_key_secret = ACCESS_KEY_SECRET
    # 获取当前时间（东8区）
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    # print("------------",body_params)
    result = send_request(access_key_id, access_key_secret, timestamp, http_url, url_params, body_params, method, header)
    return result

def rag(query, top_k=3, is_print=True):
    # online config
    QUESTION_RAG = "https://openai.100tal.com/aitext/vector-databse-search/correcting/query"
    body={
        "question":query,
        "top_k":top_k,
        # "database":[7,20]
        "database":[9,10]
    }
    res = my_request(QUESTION_RAG, {}, body, "POST")
    if is_print:
        print(res)
    return res


if __name__ == '__main__':
    import argparse
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--ocr_column", type=str, default="ocr", help="ocr column name")
    
    args = parser.parse_args()
    input_path = args.input_path
    save_path = args.save_path

    logger.info("input_path: %s", input_path)
    if not os.path.exists(input_path):
        raise Exception("file:{} not found".format(input_path))
    if input_path.endswith(".csv"):
        data = load_csv_2_dict(input_path)
    elif input_path.endswith(".xlsx"):
        import pandas as pd
        df = pd.read_excel(input_path)
        data = df.to_dict(orient='records')
    else:
        raise Exception("file:{} not found".format(input_path))

    targets = []
    logger.info("query online rag begin...")
    title_set = set()
    for idx, d in enumerate(data, start=1):

        logger.info(f"processing data {idx}/{len(data)}...")
        try:
            query = d[args.ocr_column]
        except Exception as e:
            logger.error(f"Error in getting ocr column: {e}")
            targets.append(json.dumps(t, ensure_ascii=False))
            continue

        url = d['img_url'] if 'img_url' in d else d['url']
        title = d['title'] if 'title' in d else ''
        
        t = {}
        t["url"] = url
        t["input_content"] = query
        t["rag"] = []

        if title not in title_set:
            title_set.add(title)
        else:
            query = query.replace(title, '')

        try:    
            pattern = r"##.*?##"
            query = re.sub(pattern, "", query) ## 去除手写体，仅保留题干
        except Exception as e:
            logger.error("Error in regex substitution: %s", e)
            query = ""

        if query == "":
            targets.append(json.dumps(t, ensure_ascii=False))
            continue

        cnts = 0
        while True:
            res = rag(query, top_k=args.top_k, is_print=False)
            cnts += 1
            if cnts > 10:
                break
            if res["code"] == 20000:
                rag_ = res["data"]["result"]
                break
            else:
                logger.error(f"{cnts}-th rag request failed, error message: {res}, sleep 1s and retry...")
                time.sleep(1)
        t = {}
        t["url"] = url
        t["input_content"] = query
        t["rag"] = rag_
        targets.append(json.dumps(t, ensure_ascii=False))
        time.sleep(1)

    with open(save_path, "w", encoding="utf-8", newline="") as f:
        # w=csv.writer(f)
        # w.writerow(["processed_image_url", "input_content", "rag"])
        # w.writerows(targets)
        for d in targets:
            f.write(d + '\n')
    logger.info("query online rag end...")

