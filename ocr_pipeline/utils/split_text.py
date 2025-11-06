import re

def count_kongshu(s):
    pattern = r'##(.*?)##'
    matches = re.findall(pattern, s)
    return len(matches)

def split_ocr(ocr, max_kongshu=20):
    merged_list = merge_marked_elements(ocr)
    temp_str = ""
    sum=0
    result = []
    overlap = ""
    for idx, i in enumerate(merged_list):
        # if not temp_str:
        #     temp_str = i
        sum += count_kongshu(i)
        if sum > max_kongshu and '##' in temp_str:
            result.append(temp_str)
            temp_str = merged_list[idx-1]
            sum = count_kongshu(i)
        temp_str += i
    if temp_str:
        result.append(temp_str)
    return result

def merge_marked_elements(s):
    ### 分离手写体
    # res = re.split(r'(\##[^#]+\##)', s) ### 有bug，替换成下面的函数split_hand_text @0306
    res = split_hand_text(s)
    input_list = [i for i in res if i.strip()]
    ### 合并相邻的手写体
    output_list = []
    temp_str = ""
    for item in input_list:
        # 判断是否仅包含特殊字符
        if not item.strip()== "####":
            pattern = re.compile(r'^[\W_]+$')
            is_special_str = bool(pattern.match(item))
            if is_special_str:
                continue
        # item = re.sub(r'[^\w\s]', '', item)
        if '##' in item:
            temp_str += item
        else:
            # item = re.sub(r'[^\w\s]', '', item)
            # 如果临时字符串不为空，先将其添加到输出列表中
            if temp_str:
                output_list.append(temp_str)
                temp_str = ""
            # 直接添加当前项
            output_list.append(item)   
    # 检查最后一次合并
    if temp_str:
        output_list.append(temp_str)   
    return output_list

def split_hand_text(s):
    # 定义正则表达式以匹配被“##”包含的字符串
    pattern = re.compile(r'##.*?##')
    result = []
    last_end = 0
    # 迭代字符串进行匹配
    for match in pattern.finditer(s):
        ### 找到每个手写体的起始位置和结束位置 match.start(), match.end()
        if match.start() != last_end:
            # 添加上一个匹配结束位置到当前匹配开始位置之间的字符串，印刷体部分
            result.append(s[last_end:match.start()])

        # 添加当前匹配的手写体字符串(带有##标记)
        result.append(match.group())   

        # 更新上一个匹配结束的位置
        last_end = match.end()
        
    if last_end != len(s):
        result.append(s[last_end:])
    return result