import ast
import time
from io import BytesIO
import base64
import json
import uuid
import hmac
from hashlib import sha1
import requests
import csv
import os
import shutil
from pathlib import Path
import urllib.request
import re
import sys
import os
import csv
import difflib
from Levenshtein import distance

def clean_text(text):
    text = remove_latex_commands(text)
    text = remove_punctuation_and_special_chars(text)
    text = text.replace(' ', '')
    text = re.sub(r'[。，、：；？！《》“”‘’]', '', text)
    return text

def remove_latex_commands(latex_str):
    latex_str = latex_str.replace(' ','')
    # 这个正则表达式将匹配形如 \command{...} 的 LaTeX 命令，并递归处理内部内容
    pattern = re.compile(r'\\[a-zA-Z]+\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}')
    while True:
        # 替换匹配到的内容，并递归处理内部内容
        new_str = re.sub(pattern, lambda m: m.group(1), latex_str)
        if new_str == latex_str:
            break
        latex_str = new_str
    # 最终移除多余的花括号
    text = re.sub(r'[{}]', '', latex_str)
    return text

def remove_punctuation_and_special_chars(text):
    # 使用正则表达式匹配所有非汉字、非英文字母和非数字的字符并替换为空字符串
    special_rules = [
                        ("①", "1"),
                        ("②", "2"),
                        ("③", "3"),
                        ("④", "4"),
                        ("⑤", "5"),
                        ("⑥", "6"),
                        ("⑦", "7"),
                        ("⑧", "8"),
                        ("⑨", "9"),
                        ('⑩', '10'),
                        ('⑪', '11'),
                        ('⑫', '12'),
                        ('⑬', '13'),
                        ('⑭', '14'),
                        ('⑮', '15'),
                        ("×", "错"),
                        ("✓", "对"),
                        ("<", "小于"),
                        (">", "大于")
                    ]
    for i in special_rules:
        text = text.replace(i[0], i[1])
    return re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303F]', '', text)

def normalized_levenshtein_distance(s1, s2):
    s1 = clean_punctuation(s1.strip())
    s2 = clean_punctuation(s2.strip())
    dis = distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0  # 两个字符串都是空字符串时返回0
    return dis / max_len

def lev_score(s1, s2):
    return 1-normalized_levenshtein_distance(s1, s2)

def load_csv_2_dict(csv_path):
    """加载csv为字典"""
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

def load_json_2_csv(input_json, output_csv):
    if isinstance(input_json, str) and os.path.exists(input_json):
        data = load_json(input_json)
    else:
        data = input_json
        
    # 获取字典中的所有键作为列名
    all_keys = set()
    for row in data:
        all_keys.update(row.keys())
    
    fieldnames = data[0].keys() if isinstance(data, list) else data.keys()
    fieldnames = list(fieldnames)
    new_keys = [i for i in all_keys if i not in fieldnames]
    fieldnames.extend(new_keys)

    new_dict = {key: None for key in fieldnames}
    fieldnames = new_dict.keys()
    
    # 写入 CSV 文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        if isinstance(data, list):
            for row in data:
                writer.writerow(row)
        else:
            writer.writerow(data)


def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        data=list(csv.reader(f))
    return data

def save_jsonL(save_path, data):
    """保存为jsonl"""
    with open(save_path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

def load_jsonL(json_path):
    """加载jsonl"""
    with open(json_path, 'r') as f:
        dataset = f.readlines()
    data = [json.loads(d.strip()) for d in dataset]
    return data

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data
    
def save_json(save_path, data):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def img_to_base64(image_path):
    with open(image_path,"rb") as f:
        b64_data = base64.b64encode(f.read()).decode()
    return b64_data

def text_similarity_ratio(s1,s2):
    return difflib.SequenceMatcher(s1,s2).quick_ratio()

def clean_punctuation(s):
    return re.sub(r'[^\w\s\u4e00-\u9fff]', '', s)

def url_format(params):
        """
        # 对params进行format
        # 对 params key 进行从小到大排序
        :param params: dict()
        :return:
        a=b&c=d
        """
        sorted_parameters = sorted(params.items(), key=lambda d: d[0], reverse=False)
 
        param_list = ["{}={}".format(key, value) for key, value in sorted_parameters]
 
        string_to_sign = '&'.join(param_list)
        return string_to_sign



def base2img(base64_string, img_name):
    '''
    将base64转换为jpg格式
    '''
    image_data = base64.b64decode(base64_string)
    with open(img_name, 'wb') as f:
        f.write(image_data)

def convert_to_bool(s):
    '''
    将输入字符转换为逻辑变量true/false
    '''
    if s.lower() in ['true', '1', 't', 'y', 'yes']:
        boolean_value = True
    elif s.lower() in ['false', '0', 'f', 'n', 'no']:
        boolean_value = False
    else:
        raise TypeError('Invalid value for boolean. Use True/False or equivalent.')
        # print("Invalid value for boolean. Use True/False or equivalent.")
    return boolean_value


def download_image(url, save_path):
    try:
        urllib.request.urlretrieve(url, save_path)
        # print(f"图片成功保存在 {save_path}")
    except Exception as e:
        print(f"图片{url}下载失败: {e}")

