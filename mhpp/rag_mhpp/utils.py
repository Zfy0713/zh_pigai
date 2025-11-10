import ast
import time
from io import BytesIO
import base64
import json
import uuid
import hmac
from hashlib import sha1
import uuid
from urllib import parse
import json
from urllib.parse import quote
# from attrs import field
import requests
import csv
import os
import shutil
from pathlib import Path
import urllib.request
import re
import sys
from urllib.parse import quote
import os
import csv
import json, difflib
from Levenshtein import distance

def load_jsonL(json_path):
    """加载jsonl"""
    with open(json_path, 'r') as f:
        dataset = f.readlines()
    data = [json.loads(d.strip()) for d in dataset]
    return data


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