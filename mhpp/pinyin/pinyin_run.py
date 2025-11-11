import json, csv, os, sys
import copy
import re
from time import sleep
import time
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from .Ciku import Ciku
from .utils import clean_text, mhpp, load_jsonL, save_jsonL
from Levenshtein import distance


def extract_pinyin(text):
    """
    去除字符串中的中文和标点符号
    """
    # 匹配中文字符的正则表达式：[\u4e00-\u9fff]
    # 匹配中文标点符号的正则表达式：[\u3000-\u303f\uff00-\uffef]
    pattern = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]')
    text = pattern.sub('', text)
    ## 匹配英文标点符号的正则表达式
    pattern = re.compile(r'[^\w\s]')
    text = pattern.sub('', text)
    ## 匹配数字的正则表达式
    pattern = re.compile(r'\d')
    text = pattern.sub('', text)
    return text

def pigai_ciku(ciku_path, data, remove_tones=True):
    """
    批改拼音题框补充做答结果
    ciku_path: 拼音字典路径（词库）
    data: 处理后的数据 e.g. [{"hand_text": "##小径##", "text": "xiǎo jìng", "hand_bbox": [127, 1016, 320, 1106], "topN": {"results": [{"text": " 小 径", "prob_reg": 0.99999434}, {"text": " 小 经", "prob_reg": 0.0022569296}, {"text": " 小 征", "prob_reg": 0.0016512049}, {"text": " 小 行", "prob_reg": 0.0012222874}, {"text": " 小 轻", "prob_reg": 0.00097449246}]}}, ...]
    """
    if not os.path.exists(ciku_path):
        raise FileNotFoundError(f"Ciku file not found: {ciku_path}")
    
    # 加载拼音字典
    ciku = Ciku(ciku_path)
    pinyin_dict = ciku.pinyin_dict
    if remove_tones:
        new_dict = {}
        # 去除声调和空格
        for k, v in pinyin_dict.items():
            removed_pinyin = ciku._remove_tones(k)
            removed_pinyin = removed_pinyin.replace(' ', '')
            if removed_pinyin not in new_dict:
                new_dict[removed_pinyin] = v
            else:
                # new_dict[removed_pinyin] = [new_dict[removed_pinyin]] + v
                new_dict[removed_pinyin].extend(v)
        pinyin_dict = new_dict

    ### 查询词库汉字
    matched_boxes = []
    for id, item in enumerate(data):
        img_url = item['img_url']

        hanzi_list = []
        hand_bbox = item.get('hand_bbox', None)
        if not hand_bbox:
            continue
        [x1,y1, x2, y2] = hand_bbox
        pinyin_str = item['text']
        pinyin_str = extract_pinyin(pinyin_str)  ### 提取拼音
        pinyin_str = pinyin_str.replace(' ', '')
        hand_text = item.get('hand_text', None)
        topN = item.get('topN', None)
        if pinyin_str and hand_text:
            pinyin_str = ciku._remove_tones(pinyin_str) if remove_tones else pinyin_str
            if pinyin_str in pinyin_dict:
                hanzi_list = pinyin_dict[pinyin_str]
            else:
                hanzi_list = []
                print(f"{img_url}: 拼音 {pinyin_str} 不在字典中")
        if topN:
            topN_threshold = 0.5
            topN_res = [clean_text(hand_text.replace('##',''))] + [clean_text(j['text']) for j in topN['results'] if j['prob_reg']>=topN_threshold]
        else:
            topN_res = [clean_text(hand_text.replace('##',''))]
        if len(hanzi_list) == 0:
            pigai_res = 9
            continue
        else:
            print(f"Processing for {img_url} at index {id}:\n pinyin_str: {pinyin_str},\n hanzi_list: {hanzi_list},\n topN_res: {topN_res}")
            pigai_res = any(mhpp(i,j) for i in hanzi_list for j in topN_res)
            print(f"pigai_res: {pigai_res}")
            print("="*20)

        matched_boxes.append({
            "x": x1 + (x2-x1)//2,
            "y": y1 + (y2-y1)//2,
            "result": int(pigai_res),
            "box": hand_bbox
        })

    draw_output = {
        "img_url": data[0]['img_url'],
        "matched_boxes": matched_boxes
    }
    return draw_output

def main(args):
    pinyin_input = args.pinyin_supp_ocr
    save_path = args.save_path

    data = load_jsonL(pinyin_input)

    ciku_path='/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v2/pinyin_large/pinyin_dict.json'
    draw_output = pigai_ciku(ciku_path=ciku_path, data=data, remove_tones=True)
    print(draw_output)

    if save_path.endswith('.jsonl'):
        with open(save_path, 'w') as f:
            json.dump(draw_output, f, ensure_ascii=False, indent=2)
        print(f"pinyin pigai output saved to {save_path}")
        with open(save_path.replace('.jsonl', '.csv'), 'w', encoding='utf-8-sig', newline="") as f:
            w=csv.writer(f)
            w.writerow(list(draw_output.values()))
        
    else:
        raise TypeError(f"save_path must end with 'jsonl', but got {save_path}")

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pinyin pigai")
    parser.add_argument("--ciku_path", type=str, default="/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v2/pinyin_large/pinyin_dict.json")
    parser.add_argument("--pinyin_supp_ocr", type=str, required=True, help="")
    parser.add_argument("--save_path", type=str, required=True, help="")
    args = parser.parse_args()
    main(args)