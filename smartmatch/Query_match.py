'''
页内题目匹配
'''

import json, csv
import re
import os
from Levenshtein import distance
from tqdm import tqdm
from ..utils.utils import load_jsonL


class QueryMatch:
    def __init__(self, query: str, candidate_data: dict):
        '''
        query: str, 用户查询的题目文本
        candidate_data: dict, 候选题目数据，包含题目文本和相关信息 smartmatch 传入的格式，
        例如:
        '''
        self.query = query
        self.candidate_data = candidate_data
    
    def text_similarity(self, text1: str, text2: str) -> float:
        '''
        计算两个文本的相似度，返回值在0到1之间，1表示完全相同
        '''
        if not text1 or not text2:
            return 0.0
        dist = distance(text1, text2)
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        similarity = 1 - dist / max_len
        return similarity
    
    
    def find_best_match(self) -> dict:
        '''
        在候选题目中找到与查询题目最相似的题目，返回匹配结果和相似度
        '''
        best_match = None
        highest_similarity = 0.0
        
        all_list = self.candidate_data.get('main_question_list', []) + self.candidate_data.get('sub_question_list', [])
        for candidate in all_list:
            candidate_text = candidate.get('text', '')
            sim = self.text_similarity(self.query, candidate_text)
            if sim > highest_similarity:
                highest_similarity = sim
                best_match = candidate

        if best_match:
            matched_text = best_match.get('text', '')
            ht_answer_tile_converted = best_match.get('ht_answer_tile_converted', [])
            ht_answer_text = [ans['ans_text'] for ans in ht_answer_tile_converted]

            ht_answer_text = [re.split(r'\s+', ans) for ans in ht_answer_text]
            type_list = best_match.get('type_list', [])

        return {
            'query': self.query,
            'matched_text': matched_text if best_match else '',
            'ht_answer_text': ht_answer_text if best_match else [],
            'type_list': type_list if best_match else [],
            'similarity': highest_similarity
        }
    
def single_run(item: list, url_index, query_index, args):

    url = item[url_index]
    query = item[query_index]
    pattern = r"##(.*?)##"
    query = re.sub(pattern, '', query).strip()  # 去除##之间的内容
    img_name = os.path.split(url)[-1]
    # match_dir = '/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/jiaofu/db_data/0822-2/smartmatch/smart_match_json_processed4'
    match_dir = args.smartmatch_dir

    match_file = os.path.join(match_dir, img_name.replace('.jpg', '.json'))
    candidate_data = json.load(open(match_file, 'r', encoding='utf-8'))
    bookid = candidate_data.get('book_id', '')
    page_num = candidate_data.get('page_num', '')
    query_match = QueryMatch(query, candidate_data).find_best_match()

    ## 添加书籍ID和页码信息
    query_match['book_id'] = bookid
    query_match['page_num'] = page_num
    query_match['is_in_bookid_list'] = bookid in BOOK_ID_list

    item.append(json.dumps(query_match, ensure_ascii=False, indent=2))
    item.append(bookid in BOOK_ID_list)

    return item

def main(args):
    # input_file='/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/jiaofu/db_data/0822-2/merge_vlocr/merged_ocr_supp_results.csv'
    # output_file='/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/jiaofu/db_data/0822-2/vlocr/VL_res_rag_within_page_0829.csv'

    input_chaiti_file = args.input_chaiti_file
    input_search_file = args.input_search_file
    save_path = args.output_file

    data = load_jsonL(input_chaiti_file)
    candidate_data = json.load(open(input_search_file, 'r', encoding='utf-8'))
    bookid = candidate_data.get('book_id', '')
    page_num = candidate_data.get('page_num', '')

    for item in data:
        url = item["img_url"]
        query = item["text_vl"]
        pattern = r"##(.*?)##"
        query = re.sub(pattern, '', query).strip()  # 去除##之间的内容
        img_name = os.path.split(url)[-1]
        query_match = QueryMatch(query, candidate_data).find_best_match()
        query_match['book_id'] = bookid
        query_match['page_num'] = page_num
        item['find_within_page'] = query_match
    
    with open(save_path, 'w') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_chaiti_file', type=str, default='')
    argparser.add_argument('--input_search_file', type=str, default='')
    argparser.add_argument('--output_file', type=str, default='')
    args = argparser.parse_args()
    main(args)

