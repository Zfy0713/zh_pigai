'''线上9、10库+图库批改链路'''

import numpy as np
import pandas as pd
from tqdm import tqdm
from Levenshtein import distance
import json, csv, os, sys, re, ast
import threading
from .utils import remove_latex_commands, remove_punctuation_and_special_chars, lev_score, load_jsonL
from .answer_extract import answer_extract

os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

def is_choice(text):
    if len(text)>1:
        return False
    return text.upper() in ['A','B','C','D','E','F','G','H']

def is_number(text):
    return text.isdigit()

def is_pinyin_str(s):
    # 拼音字母和带声调的拼音字母的范围
    pinyin_with_tones = "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ"
    pinyin_chars = f"a-zA-Z{pinyin_with_tones}"
    pattern = re.compile(rf'^[{pinyin_chars}]+$')
    return bool(pattern.match(s))

def is_subsequence(sub, main):
    sub_len = len(sub)
    main_len = len(main)
    if sub_len == 0:
        return True 
    sub_index = 0
    for char in main:
        if char == sub[sub_index]:
            sub_index += 1
            if sub_index == sub_len:
                return True
    return False

def clean_text(text):
    text = remove_latex_commands(text)
    text = remove_punctuation_and_special_chars(text)
    text = text.replace(' ', '')
    text = re.sub(r'[。，、：；？！《》“”‘’]', '', text)
    return text

def longest_common_sublist(s1, s2, is_pinyin):
    '''
    最长公共子列表序列
    '''
    m = len(s1)
    n = len(s2)
    
    L = [[0 for x in range(n+1)] for y in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif _rag_mhpp_topN(s1[i-1], s2[j-1], is_pinyin) == 1:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    # L[m][n] 存储的是 s1 和 s2 的 LCS 长度
    index = L[m][n]
    lcs = []
    common_pigai_res = []
    ### 存储公共元素在原列表中的位置
    position1 = []
    position2 = []

    # 从 L[m][n] 开始逆向构建 LCS
    i = m
    j = n
    while i > 0 and j > 0:
        # if s1[i-1] == s2[j-1]:
        if (_rag_mhpp_topN(s1[i-1], s2[j-1], is_pinyin) == 1):
            lcs.append(s2[j-1])
            common_pigai_res.append(_rag_mhpp_topN(s1[i-1], s2[j-1], is_pinyin))
            position1.append(i-1)
            position2.append(j-1)
            i -= 1
            j -= 1
            index -= 1
        elif L[i-1][j] > L[i][j-1]:
            i -= 1
        else:
            j -= 1
    lcs.reverse()
    common_pigai_res.reverse()
    position1.reverse()
    position2.reverse()

    # return "".join(lcs)
    return lcs, common_pigai_res, position1, position2

def most_common(col):
    stat_res = {}
    for k in col: ### 统计批改结果和出现次数
        stat_res[k] = stat_res.get(k,0) + 1
    if 1 in stat_res:
        return 1
    elif 0 in stat_res:
        return 0
    else:
        return 9

def _rag_mhpp_topN(hand_text_info, answer: str, is_pinyin) -> int:
    '''
    单个手写体topN匹配
    '''
    if not hand_text_info['topN']:
        topN_res = [clean_text(hand_text_info['text'])]
    else:
        topN_res = [clean_text(hand_text_info['text'])] + [clean_text(j['text']) for j in hand_text_info['topN']['results'][:2] if j['prob_reg']>=0.5] ### topN取前两个置信度大于0.5
        topN_res += [clean_text(j['text']) for j in hand_text_info['topN']['results'][2:] if j['prob_reg'] >= 0.75] ### 后三个topN结果要求置信度大于0.75
        topN_res = list(dict.fromkeys(topN_res)) ###去重

    temp_pigai_result = list(map(lambda x:_rag_mhpp(x, answer, is_pinyin), topN_res))
    
    topN_pigai_res = {}
    for k in temp_pigai_result: ### 统计批改结果和出现次数
        topN_pigai_res[k] = topN_pigai_res.get(k,0) + 1
    if 1 in topN_pigai_res:
        return 1
    elif 0 in topN_pigai_res:
        return 0
    else:
        return 9
    
    # return most_common(temp_pigai_result)

def _rag_mhpp(hand_text, answer, is_pinyin): ### 字符串级别模糊匹配
    hand_text = clean_text(hand_text)
    answer = clean_text(answer)
    len_answer = len(answer)
    len_hand = len(hand_text)
    dis = distance(answer, hand_text)
    if is_choice(hand_text):
        if hand_text!=answer:
            return 0
        else:
            return 1
    if len_hand>=7:
        return 1
    if hand_text == answer:
        return 1
    if len(hand_text) == 1 and hand_text in answer:
        return 1
    if len(hand_text) == 1 and hand_text != answer:
        return 0
    if len(hand_text) == 2 and dis >= 1:
        return 0

    if (is_pinyin_str(hand_text) or is_pinyin) and dis < 2:
        return 1
    
    return 9

def rag_mhpp(hand_text_list, answer_list, is_pinyin):
    '''
    hand_text_list: 手写体列表(有topN字段)
    answer_list: 题库答案列表
    return: 手写体批改结果，没有则为空
    '''

    pigai_result = [9 for _ in range(len(hand_text_list))] ## 每个手写体对应一个批改结果

    not_matched_answer = []
    ### 选择题/序号题 且题库答案和手写体作答个数相同，按照位置批改
    if (all([is_choice(clean_text(i['text'])) for i in hand_text_list]) or all([is_number(clean_text(i['text'])) for i in hand_text_list])) and len(hand_text_list) == len(answer_list):
    # if len(hand_text_list) == len(answer_list):
        for idx in range(len(hand_text_list)):
            pigai_result[idx] = _rag_mhpp_topN(hand_text_list[idx], answer_list[idx], is_pinyin)
            if pigai_result[idx] == 9:
                not_matched_answer.append(answer_list[idx])

    ### 
    else:
        lcs, common_pigai_res, pos1, pos2 = longest_common_sublist(hand_text_list, answer_list, is_pinyin)
        ### pos1: 手写体匹配位置
        for idx in pos1:
            pigai_result[idx] = common_pigai_res.pop(0)

        for i in range(1, len(pos1)):
            diff = pos1[i] - pos1[i-1]
            if diff > 1:
                diff2 = pos2[i] - pos2[i-1]
                if diff == diff2:
                    for j in range(pos1[i-1]+1, pos1[i]):
                        pigai_result[j] = 0
    
        
        for idx in range(len(answer_list)):
            if idx not in pos2: ### 未匹配到的答案
                not_matched_answer.append(answer_list[idx])

    # not_pigai_handtext = []
    for idx in range(len(pigai_result)):
        if pigai_result[idx] == 9: ### 未批改的手写体作答再次匹配
            
            if not hand_text_list[idx]['topN']:
                topN_res = [clean_text(hand_text_list[idx]['text'])]
            else:
                topN_res = [clean_text(hand_text_list[idx]['text'])] + [clean_text(j['text']) for j in hand_text_list[idx]['topN']['results'][:2] if j['prob_reg']>=0.5] ### topN取前两个
                topN_res += [clean_text(j['text']) for j in hand_text_list[idx]['topN']['results'][2:] if j['prob_reg'] >= 0.75] ### 后三个topN结果要求置信度大于0.3
                topN_res = list(dict.fromkeys(topN_res))
            for ans in not_matched_answer[:]:
                if any([clean_text(j) in ans for j in topN_res]):
                    not_matched_answer.remove(ans)
                    pigai_result[idx] = 1
                    break

    return pigai_result

def rag_pigai(item, is_tusou=False):

    # ocr = item['ocr'].strip()
    ocr = item['text_vl'].strip()
    title = item['title'].strip()
    hand_text_list = ast.literal_eval(item['hand_text_list']) if isinstance(item['hand_text_list'], str) else item['hand_text_list'] ## 学生作答

    len_hand = len(re.findall(r"##.*?##", ocr)) ### 手写体个数
    if len_hand == 0:
        return
    ocr = re.sub(r'##.*?##', '', ocr)
    if ocr.startswith(title):
        ocr = ocr[len(title):].strip()

    pinyin_keywords = ['看拼音','读拼音', '读音']
    is_pinyin = any([keyword in title for keyword in pinyin_keywords])
    item['is_pinyin'] = is_pinyin


    moxie_keywords = ['默写', '课文背诵', '日积月累', '古诗词积累', '诗词积累', '名句积累', '古诗积累', '必背', '经典诗词']
    if any(keyword in title for keyword in moxie_keywords) and not is_pinyin:
        is_moxie = 1
    else:
        is_moxie = 0
    item['is_moxie'] = is_moxie
    

    # title_keywords = ['句子', '句式', '排序', '组词', '写词语', '阅读']
    # ocr_keywords = title_keywords + ['近义词','反义词',]
    # if any(keyword in ocr for keyword in ocr_keywords):
    #     return
    # if any(keyword in title for keyword in title_keywords):
    #     return

    bukepi_keywords = ['表格', '连一连','连线', '划去','笔顺','笔画',] ### 不可批题型
    pidui_keywords = ['大写字母', '小写字母'] ### 批对题型
    if any(keyword in title for keyword in bukepi_keywords + pidui_keywords) or any(keyword in ocr for keyword in bukepi_keywords + pidui_keywords):
        is_pidui = 1
    else:
        is_pidui = 0
    if is_moxie:
        ### 默写题都判对
        result = [1 for _ in range(len(hand_text_list))]

    elif is_pidui:
        result = [1 for _ in range(len(hand_text_list))]
    else:
        ### 答案抽取

        rag_res = answer_extract(item) if 'rag_answer' not in item else item['rag_answer']
        # rag_res = answer_extract(item)
        if not rag_res: ### 不使用题库
            return
        if isinstance(rag_res, str):
            rag_res = ast.literal_eval(rag_res)
        rag_answer = rag_res ## 题库答案抽取结果
        item['rag_answer'] = rag_res

        pattern = r'^[0-9零一二三四五六七八九]+'
        rag_answer = [re.sub(pattern, '', i) for i in rag_answer]
        def remove_spaces_inside_dollar(s): ### 去掉$$...$$中的空格
            return re.sub(r'\$\$(.*?)\$\$', lambda match: '$$' + re.sub(r'\s+', '', match.group(1)) + '$$', s)
        
        rag_answer = [remove_spaces_inside_dollar(i) for i in rag_answer] ### 删除题库答案中$$...$$中的空格
        
        is_xuhao = False
        if '序号' in title:
            is_xuhao = True
        if is_xuhao: ## 序号题中带圈的数字 “①②③④⑤⑥……” 不作为题号分割
            pattern = r'【\d+】|[\d+]|（\d+）|\d+\.|\(\d+\)|\d+|；|;|、|，|。|！|？|\n| |【小题\d+】|\[小题\d+\]|【示例】|[示例]|<示例>|\$\$' ### 按照数字、括号数字、分号、换行符分割题库答案
        else:
            pattern = r'【\d+】|[\d+]|（\d+）|\d+\.|\(\d+\)|\d+|；|;|、|，|。|！|？|\n| |【小题\d+】|\[小题\d+\]|【示例】|[示例]|<示例>|[\u2460-\u2473]|\$\$' ### 按照数字、括号数字、分号、换行符分割题库答案
        
        rag_answer_list = [re.split(pattern, i) for i in rag_answer]
        rag_answer_list = [[remove_latex_commands(j) for j in i] for i in rag_answer_list]
        rag_split_res = []
        for idx, i in enumerate(rag_answer_list):
            ii = [j.replace('✓','') for j in i]
            rag_split_res.append([clean_text(x) for x in ii if clean_text(x)])

        if len(rag_answer) == 0:
            return
        item['rag_split_res'] = rag_split_res
        ### 模糊匹配mhpp-题库批改
        pigai_res = [rag_mhpp(hand_text_list, rag, is_pinyin) for rag in rag_split_res]
        pigai_res = np.array(pigai_res)
        try:
            result = np.apply_along_axis(most_common, axis=0, arr=pigai_res)
            result = result.tolist()

        except ValueError as e:
            print("Error:", e)
            print(pigai_res)
            return
    

    ### 储存最终题库批改结果
    matched_box = []
    for i, res in zip(hand_text_list, result):
        hand_text = i['text'].strip('##')

        x1, y1 = i['x'], i['y']
        width = i['width']
        height = i['height']
        x2, y2 = x1 + width, y1 + height

        if len(hand_text) >= 7:
            res = 1
        
        # ### 题库未批改且手写体长度>=2，都批对
        # if res == 9 and len(hand_text) >= 2:
        #     res = 1
        
        matched_box.append({
            "x": x1 + (x2-x1)//2,
            "y": y1 + (y2-y1)//2,
            "result": res,
            "box": [x1, y1, x2, y2]

            # "matched_boxes": [
            #     {
            #         "x": i['x'], "y": i['y'], "width": i['width'], "height": i['height'],'text': i['text']
            #     }
            # ]
        })

    item['matched_box'] = matched_box

    return {
        "img_url": item['img_url'],
        "matched_box": matched_box
    }


def main(args):
    from collections import defaultdict
    from tqdm import tqdm
    input_path = args.input_path
    save_path = args.save_path

    data = load_jsonL(input_path)
    draw_output = defaultdict(list)
    for item in tqdm(data):
        img_url = item['img_url']
        output = rag_pigai(item)
        if output:
            matched_box = output['matched_box']
        else:
            matched_box = []

        if img_url in draw_output:
            draw_output[img_url] += matched_box
        else:
            draw_output[img_url] = matched_box
    
    outputs = []
    for img_url, matched_box in draw_output.items():
        outputs.append({
            "img_url": img_url,
            "matched_box": matched_box
        })



    if save_path.endswith('.jsonl'):
        with open(save_path, 'w') as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
        print(f"rag pigai results saved to {save_path}")
        with open(save_path.replace('.jsonl', '.csv'), 'w', encoding='utf-8-sig', newline="") as f:
            w=csv.writer(f)
            for d in outputs:
                w.writerow([d['img_url'], d['matched_box']])
        
    else:
        raise TypeError(f"save_path must end with 'jsonl', but got {save_path}")

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="tiku pigai")
    parser.add_argument("--input_path", type=str, required=True, help="")
    parser.add_argument("--save_path", type=str, required=True, help="")

    args = parser.parse_args()
    main(args)


