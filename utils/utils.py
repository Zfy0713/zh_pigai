import json
import csv
import difflib
import re

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

def load_csv(path):
    csv.field_size_limit(100000000)
    with open(path, 'r', encoding='utf-8-sig') as f:
        data=list(csv.reader(f))
    return data

def text_similarity_ratio(s1,s2):
    return difflib.SequenceMatcher(s1,s2).quick_ratio()

def clean_punctuation(s):
    return re.sub(r'[^\w\s\u4e00-\u9fff]', '', s)

def save_json_2_csv(input_json, output_csv):
    if isinstance(input_json, str) and os.path.exists(input_json):
        data = load_json(input_json)
    else:
        data = input_json
    # 获取字典中的所有键作为列名
    fieldnames = data[0].keys() if isinstance(data, list) else data.keys()
    # 写入 CSV 文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        if isinstance(data, list):
            for row in data:
                writer.writerow(row)
        else:
            writer.writerow(data)

def flatten_nested_list(nested_list):
    """
    将任意深度的嵌套列表转换为一维列表
    
    参数:
    nested_list: 可能包含多层嵌套的列表
    
    返回:
    完全展平的一维列表
    """
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            # 递归处理嵌套列表
            flattened.extend(flatten_nested_list(item))
        else:
            # 非列表元素直接添加
            flattened.append(item)
    return flattened