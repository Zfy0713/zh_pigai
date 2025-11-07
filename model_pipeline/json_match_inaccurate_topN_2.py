import json, csv, os, sys
import itertools, re, difflib
from Levenshtein import distance
from ..utils.utils import load_json, load_csv, text_similarity_ratio, clean_punctuation
import ast

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

# def remove_latex_commands(latex_str):
#     latex_str = latex_str.replace(' ','')
#     # 使用正则表达式移除所有的LaTeX命令
#     text = re.sub(r'\\[a-zA-Z]+\{(.*?)\}', r'\1', latex_str)
#     # 去除多余的花括号
#     text = re.sub(r'[{}]', '', text)
#     return text

def contain_alphabeta(s: str):
    return any(char.isalpha() for char in s)

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
    return re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)

def remove_until_triple_backtick(s):
    if "```" in s:
        while not s.endswith("```") and len(s) > 0:
            s = s[:-1]
    return s

def remove_comma_after_bracket(s):
    # 使用正则表达式匹配并替换
    s = re.sub(r'\],\s*}', ']}', s)
    return s

def remove_after_last_char(s, c):
    last_brace_index = s.rfind(c)
    
    if last_brace_index != -1:
        return s[:last_brace_index + 1]
    else:
        return s
    
def remove_text_before_brace(s):
    index = s.find("{")
    if index != -1:
        return s[index:]
    return s

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

def extract_json(json_str):
    sys.set_int_max_str_digits(0)
    json_str = remove_until_triple_backtick(json_str)
    json_str = remove_comma_after_bracket(json_str)
    json_str = json_str.strip("```")
    json_str = json_str.rstrip("```\n")
    json_str = json_str.strip("json")
    json_str=json_str.replace("\\underline{}", "")
    json_str=json_str.replace("\\overline{}", "")
    try:
        # 尝试解析 JSON 字符串
        parsed_json = json.loads(json_str)
        # print("JSON is valid.")
        return parsed_json
    except json.JSONDecodeError as e:
        # 捕获 JSON 解析错误，并输出详细信息
        print(f"JSON decode error: {e}")
        print(f"Error at line {e.lineno}, column {e.colno}: {e.msg}")
        # 输出出现错误的部分周围的一些字符以帮助调试
        error_pos = e.pos
        start_context = max(0, error_pos - 50)
        end_context = min(len(json_str), error_pos + 50)
        print("Problematic JSON string context:")
        print(json_str[start_context:end_context])
        return {"data": []}
        # raise
    
def mhpp(hand_text, answer, question_type):
    len_answer = len(answer)
    len_hand = len(hand_text)
    dis = distance(answer, hand_text)
    # if not answer_text:
    #     continue
    if not answer:# or len(answer_truth) == 0:
       return 1
    if len(hand_text)>30:
        return 1
    if hand_text == answer:
        return 1
    if (is_subsequence(answer, hand_text) or is_subsequence(hand_text, answer)):
        if dis <3:
            return 1
        else:
            return 0
    if "选择" in question_type:
        if hand_text != answer:
            return 0
    if '拼音' in question_type:
        if dis >= 1:
            return 0
    if len_answer > 3 and dis < 3:
        return 1
    if len_answer > 5 and dis < 4:
        return 1
    if len_answer > 8 and dis < 7:
        return 1
    return 0
    

def match_inaccurate(json_resp: str, hand_text_list, coordinate):
    json_resp = remove_after_last_char(json_resp, '}')
    json_resp = remove_text_before_brace(json_resp)
    x_min = 99999
    y_min = 99999
    x_max = -99999
    y_max = -99999
    if coordinate:
        import ast
        coordinate = ast.literal_eval(coordinate)
        for i in coordinate:
            x_min = min(x_min, int(i['x']))
            x_max = max(x_max, int(i['x']))
            y_min = min(y_min, int(i['y']))
            y_max = max(y_max, int(i['y']))
    lines = json_resp.split('\n')
    filtered_lines = [line for line in lines if not line.strip().lstrip().startswith('"question"')]
    json_str = '\n'.join(filtered_lines)
    # filtered_lines = [line for line in lines if line.strip().lstrip().startswith('"question"')]
    # filtered_lines = [line[line.find(':')+3:-2] for line in filtered_lines]
    correct_json = extract_json(json_str)
    # correct_json = extract_json(json_resp)
    response_all = []
    if 'data' not in correct_json:
        print("没有data字段")
        return []
    for item in correct_json['data']:
        if 'response' in item:
            for i in item['response']:
                i['type'] = item['type'] if 'type' in item else ""
                response_all.append(i)
    flatten_singleBox = hand_text_list
    matched_results = []
    for answer_item in response_all:
        # print(answer_item)
        answer_truth = str(answer_item['answer']) if 'answer' in answer_item else ""  # 正确答案
        answer_text = str(answer_item['hand_text']) if 'hand_text' in answer_item else "" # 学生手写做答
        question_type = str(answer_item['type']) #题目类型
        result = 0 if 'result' in answer_item and answer_item['result']=="错误" else 1 #模型批改结果
        answer_truth = remove_punctuation_and_special_chars(answer_truth)
        answer_text = remove_punctuation_and_special_chars(answer_text)
        if len(answer_text)==0:
            continue ## 如果当前answer_text为空，直接跳过
        answer_text = answer_item['hand_text'] # 学生手写做答
        # 临时存储与当前answer匹配的singleBoxes
        temp_single_boxes = []
        # 在singleBox中查找匹配的text
        for box in flatten_singleBox[:]:
            # print('='*20)
            # print("current box: ", box)
            if not box['is_print'] and "##" in box['text']:
                box_text = box['text'].replace("##", "")
                clean_box_text = remove_punctuation_and_special_chars(box_text)
                clean_answer_text = remove_punctuation_and_special_chars(answer_text)

                if not clean_answer_text:
                    continue
                
                text_sim = text_similarity_ratio(clean_answer_text, clean_box_text)
                if clean_box_text and clean_box_text in clean_answer_text or text_sim>=0.70:
                    topN_list = box['topN']
                    answer_text = remove_punctuation_and_special_chars(answer_text)
                    
                    if topN_list and answer_text and answer_truth:

                        topN_res = [clean_text(answer_text)] + [clean_text(j['text']) for j in topN_list['results'][:2] if j['prob_reg']>=0.5] ### topN取前两个
                        topN_res += [clean_text(j['text']) for j in topN_list['results'][2:] if j['prob_reg'] >= 0.75] ### 后三个topN结果要求置信度大于0.3
                        topN_res = list(dict.fromkeys(topN_res))
                        # min_dis = distance(answer_text, answer_truth)
                        # min_dis = float('inf')
                        # if answer_text in topN_list:
                        #     topN_res = topN_list[answer_text]
                        # topN_res = topN_list['results']
                        # for item in topN_list['results']:
                        #     prob = item['prob_reg']
                        #     if prob < 0.3:
                        #         continue
                        #     s1 = remove_latex_commands(item['text'])
                        #     s1 = remove_punctuation_and_special_chars(s1)
                        #     cur_dis = distance(s1, answer_truth)
                        #     if cur_dis < min_dis:
                        #         min_dis = cur_dis
                        #         min_item = item
                        # if distance(answer_text, answer_truth) >= min_dis:
                        #     answer_text = remove_latex_commands(min_item['text'])
                        #     answer_text = remove_punctuation_and_special_chars(answer_text)
                        #     answer_text = answer_text.replace(' ', '')

                        if any([mhpp(i, answer_truth, question_type)==1 for i in topN_res]):
                            result = 1
                    # print('current faltten singlebox:', flatten_singleBox)

                    flatten_singleBox.remove(box)
                    x = box['x'] + box['width']//2
                    y = box['y'] + box['height']//2
                    if x <= x_min or x >= x_max or y <= y_min or y >= y_max:
                        continue
                    temp_single_boxes.append(box)
                    if clean_box_text and answer_text.endswith(clean_box_text): ### 搜到answer的末尾，结束当前answer的搜索
                        break
                elif clean_box_text and clean_answer_text in clean_box_text:
                    x = box['x'] + box['width']//2
                    y = box['y'] + box['height']//2
                    if x <= x_min or x >= x_max or y <= y_min or y >= y_max:
                        continue
                    temp_single_boxes.append(box)
                    break
        # 如果匹配到了，把它们加入到结果中
        # if len(answer_text) > 7:
        #     result = 1
            
        if temp_single_boxes:
            matched_results.append({
                'answer': answer_text,
                'result': result,
                'matched_boxes': temp_single_boxes
            })
    return matched_results

def main(args):
    model_output_path = args.model_output ### 模型批改结果
    jiaozheng_path = args.jiaozheng_path ### 原图路径
    pigai_save_path = args.pigai_save_path

    

if __name__ == "__main__":
    path = sys.argv[1]
    json_input = load_csv(path)
    json_output = []

    path = sys.argv[2]
    ocr_res = load_json(path)
    colnames= json_input[0]
    hand_text_list_index = colnames.index('hand_text_list')
    coordinate_index = colnames.index('vertices')
    url_index = colnames.index('img_url')
    for i, line in enumerate(json_input[1:]):
        # if pt >= len(json_input):
        #     continue
        # if i+1 < int(json_input[pt][0].split('_')[0]):
            # continue
        # print(i)
        # print(pt)
        img_url = line[url_index]
        # ocr = line[4]
        # testline = '二、结合自身生活经验和理解完成练习。 2. 广东真题 从下面的情境中选择一个,结合你对身边事物的感受,就心情“好”与“不好”这两种状 态分别写几句话。 ①外出研学 ②参加比赛 ③走在上学的路上 我选第( ##&①## (1)心情“好” ##外出研学的车上,窗外的风景如画,鸟语龙香,让人心旷神怡。## 项。 (2)心情“不好”, ##天黑上乌云密布##'
        # if ocr != testline:
        #     continue
        # hand_text_list = ast.literal_eval(line[-2])
        hand_text_list = ast.literal_eval(line[hand_text_list_index])
        output = []
        # while pt_new < len(json_input) and (not json_input[pt_new][0] or json_input[pt][0].split('_')[0] == json_input[pt_new][0].split('_')[0]):
        correct_response = line[-1]
        coordinate = line[coordinate_index]
        # topN_list = ast.literal_eval(line[5]) if len(json_input[pt_new][5]) > 0 else None
        if len(correct_response) < 10:
            continue
        result = match_inaccurate(correct_response, hand_text_list, coordinate) ## 模糊匹配批改结果
        for j in result:
            for k in j['matched_boxes']:
                x = k['x']+k['width']//2
                y = k['y']+k['height']//2
                x1 = k['x']+k['width']
                y1 = k['y']+k['height']
                # output.append({"x":x, "y":y, "result":j['result']})
                output.append({"x":x, "y":y, "result":j['result'], "box":[k['x'], k['y'], x1,y1]})
        json_output.append([img_url, output])

    merge_data = {}
    for i in json_output:
        if i[0] in merge_data:
            merge_data[i[0]] += i[1]
        else:
            merge_data[i[0]] = i[1]
    json_output_1 = [[url, match_boxes] for url, match_boxes in merge_data.items()]

    csv_path = sys.argv[3]
    with open(csv_path, 'w', encoding='utf-8', newline="") as f:
        w = csv.writer(f)
        w.writerows(json_output_1)
        
        