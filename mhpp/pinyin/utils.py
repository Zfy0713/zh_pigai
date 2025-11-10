import re, json, csv
from Levenshtein import distance

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

def mhpp(s1,s2):
    if s1 == s2:
        return 1
    if is_subsequence(s1,s2) or is_subsequence(s2,s1):
        return 1
    if distance(s1, s2) <= 1:
        if len(s1)<=2 and len(s2)<=2:
            return 0
        return 1
    return 0