import json


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