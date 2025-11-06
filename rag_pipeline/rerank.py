from itertools import zip_longest
import json
import re
import csv
import sys
import torch
import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from tqdm import tqdm

class Document:
    def __init__(self, content, answer, analysis, source, relevance=0.0, score=0.0):
        self.content = content
        self.answer = answer
        self.analysis = analysis
        self.source = source
        self.relevance = relevance
        self.score = score

    def __str__(self):
        return f'Document({self.content},{self.answer},{self.analysis},{self.source},{self.relevance},{self.score})'

class RankModel:

    def __init__(self, rerank_model_path, top_n, max_length=512):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")        
        self.rerank_model, self.tokenizer = self.load_reranker_model(model_path = rerank_model_path)
        self.top_n = top_n
        self.max_length = max_length

    def load_reranker_model(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        reranker_model.eval()
        # reranker_model.to(self.device)        
        return reranker_model, tokenizer

    def rerank_docs(self, docs, query):
        pairs = [[query,docs[i].content] for i in range(len(docs))]
        print(len(pairs))
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = torch.sigmoid(torch.tensor(scores)).float()
        arr = np.array(scores)
        top_n = min(self.top_n, len(docs))
        top_n_indice = arr.argsort()[-top_n:][::-1]
        new_docs = []
        print("====== scores: ",scores," =======")
        for i in top_n_indice:
            docs[i].relevance = scores[i].item()
            new_docs.append(docs[i])
        return new_docs

def json_to_dict(json_path):
    data_list =[]
    with open(json_path,'r', encoding='UTF-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            data_list.append(dic)
    return data_list


if __name__ == '__main__':
    import argparse
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank_model_path", type=str, default="/mnt/pfs_l2/jieti_team/APP/hegang/models/hegang/models/official/BAAI/bge-reranker-large")
    parser.add_argument("--bge_path", type=str)
    parser.add_argument("--tusou_path", type=str)
    parser.add_argument("--es_path", type=str, default='')
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()
    json_path_bgem3 = args.bge_path
    json_path_tusou = args.tusou_path
    json_path_es = args.es_path
    save_path = args.save_path

    data_tusou_recall = json_to_dict(json_path_tusou)
    data_bgem3_recall = json_to_dict(json_path_bgem3)
    if json_path_es == '':
        data_es_recall = {}
    else:
        data_es_recall = json_to_dict(json_path_es)

    rerank_model_path = args.rerank_model_path
    rerank_model = RankModel(rerank_model_path, top_n=args.top_k)

    print(len(data_bgem3_recall), len(data_tusou_recall))
    targets = []
    assert len(data_bgem3_recall) == len(data_tusou_recall)
    if data_es_recall:
        assert len(data_bgem3_recall) == len(data_es_recall)

    # if data_es_recall:  # 当 data_es_recall 不为空
    #     pbar = tqdm(zip(data_bgem3_recall, data_tusou_recall, data_es_recall))
    # else:  # 当 data_es_recall 为空
    #     pbar = tqdm(zip(data_bgem3_recall, data_tusou_recall))

    # 根据是否有 es 数据决定 zip 的参数
    zip_args = [data_bgem3_recall, data_tusou_recall]
    if data_es_recall:
        zip_args.append(data_es_recall)

    pbar = tqdm(zip(*zip_args))

    url_count = 0
    url_seen = {}

    for items in pbar:
        b, t = items[0], items[1]
        es = items[2] if len(items) > 2 else None

        query = b["input_content"].strip()
        ocr = b["ocr"] if "ocr" in b else ""
        url = b["url"].strip()
        if url not in url_seen:
            url_seen[url] = url_count
            url_count += 1
        if query == "" or url == "":
            targets.append([url_count, url, query, ocr, "[]"])
            continue
        pattern = r"##.*?##"
        query = re.sub(pattern, "", query) ## 去除手写体，仅保留题干            
        tusou_rag_list = t["rag"]
        bgem3_rag_list = b["rag"]["9"] # text
        bgem3_img_rag_list = b["rag"]["10"] # img
        if not es:
            es_rag_list = []
        else:
            es_rag_list = es.get('rag_list', [])
        print(len(tusou_rag_list), len(bgem3_rag_list), len(bgem3_img_rag_list), len(es_rag_list))

        if len(tusou_rag_list) == 0 and len(bgem3_rag_list) == 0 and len(bgem3_img_rag_list) == 0 and len(es_rag_list) == 0:
            targets.append([url_count, url, query, ocr, "[]"])
            print(len(tusou_rag_list), len(bgem3_rag_list),len(bgem3_img_rag_list), len(es_rag_list))
            continue

        ku_tusou_docs = [Document(item["content"], item["answer"], item["analysis"], "tusou", item["score"]) for item in tusou_rag_list]
        ku_bgem3_docs = [Document(item["content"], item["answer"], item["analysis"], "bgem3", item["score"]) for item in bgem3_rag_list]
        ku_bgem3_img_docs = [Document(item["content"], item["answer"], item["analysis"], "bgem3", item["score"]) for item in bgem3_img_rag_list]
        ku_es_docs = [Document(item["content"], item["answer"], item["analysis"], "es", item["search_score"]) for item in es_rag_list]
        docs = ku_tusou_docs + ku_bgem3_docs + ku_bgem3_img_docs + ku_es_docs
        docs = [d for d in docs if d.answer not in ['', '答案见上', '略', '答案1123','答案1 123']]
        docs = rerank_model.rerank_docs(docs, query)
        
        rag_result = []
        for idx, result in enumerate(docs):
            rag = {}
            rag["rank"] = str(idx)
            rag["similarity"] = str(result.relevance)
            rag["question"] = result.content
            rag["answer"] = result.answer
            rag["analysis"] = result.analysis
            rag["source"] = result.source
            rag_result.append(rag)
        topk_str = json.dumps(rag_result, ensure_ascii=False, indent=4)
        targets.append([url_count, url, query, ocr, topk_str])

    headline = ["num", "url","content", "ocr", "topk"]
    with open(save_path, 'w', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headline)
        for line in targets:
            w.writerow(line)