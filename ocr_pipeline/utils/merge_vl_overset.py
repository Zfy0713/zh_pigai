import re

def merge_overset(response: str) -> str:
    '''
    合并例如\overset{xiǎo xī }{\tianzige{<ans><ref>小</ref><box>[[192, 156, 250, 429]]</box></ans>}\tianzige{<ans><ref>溪</ref><box>[[249, 156, 329, 445]]</box></ans>}} 中两个tianzige的答案
    同时将两个box坐标合并
    '''
    
    # 检查是否包含 <ans></ans> 标签，如果没有则直接返回原字符串
    if '<ans>' not in response or '</ans>' not in response:
        return response
    
    import re
    
    def find_matching_brace(text, start_pos):
        """从start_pos位置的{开始，找到匹配的}的位置"""
        if start_pos >= len(text) or text[start_pos] != '{':
            return -1
        
        brace_count = 0
        for i in range(start_pos, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return i
        return -1
    
    def process_single_overset(match):
        overset_start = match.start()
        overset_text = match.group(0)
        
        # 找到 \overset{ 的位置
        overset_pos = overset_text.find('\\overset{')
        if overset_pos == -1:
            return overset_text
        
        # 找到第一个花括号的结束位置
        first_brace_start = overset_pos + len('\\overset')
        first_brace_end = find_matching_brace(overset_text, first_brace_start)
        if first_brace_end == -1:
            return overset_text
        
        # 找到第二个花括号的开始和结束位置
        second_brace_start = first_brace_end + 1
        if second_brace_start >= len(overset_text) or overset_text[second_brace_start] != '{':
            return overset_text
        
        second_brace_end = find_matching_brace(overset_text, second_brace_start)
        if second_brace_end == -1:
            return overset_text
        
        # 提取内容
        first_content = overset_text[first_brace_start + 1:first_brace_end]
        second_content = overset_text[second_brace_start + 1:second_brace_end]
        
        # 在第二个花括号内容中找到所有的 tianzige
        second_content = second_content.replace('\tianzige', '\\tianzige')
        tianzige_pattern = r'\\tianzige\{<ans><ref>(.*?)</ref><box>\[{1,2}(.*?)\]{1,2}</box></ans>\}'
        tianzige_matches = re.findall(tianzige_pattern, second_content, re.DOTALL)
        
        if len(tianzige_matches) <= 1:
            # 如果只有一个或没有 tianzige，不需要合并
            return overset_text
        
        # 合并所有的答案文本
        merged_text = ''.join([tmatch[0] for tmatch in tianzige_matches])
        
        # 合并所有的 box 坐标
        all_coords = []
        for tmatch in tianzige_matches:
            coords_str = tmatch[1]
            # 解析坐标字符串，提取数字
            coords = re.findall(r'[\d.]+', coords_str)
            if len(coords) >= 4:
                coords = [float(c) for c in coords[:4]]
                all_coords.append(coords)
        
        if all_coords:
            # 计算包围盒：取最小的x1,y1和最大的x2,y2
            min_x = min(coord[0] for coord in all_coords)
            min_y = min(coord[1] for coord in all_coords) 
            max_x = max(coord[2] for coord in all_coords)
            max_y = max(coord[3] for coord in all_coords)
            merged_box = f"[[{int(min_x)}, {int(min_y)}, {int(max_x)}, {int(max_y)}]]"
        else:
            merged_box = "[[1,1,1,1]]"
        
        # 创建合并后的单个 tianzige
        merged_tianzige = f'\\tianzige{{<ans><ref>{merged_text}</ref><box>{merged_box}</box></ans>}}'
        
        # 移除原有的所有 tianzige，用合并后的替换
        modified_content = re.sub(tianzige_pattern, '', second_content, flags=re.DOTALL)
        modified_content = merged_tianzige + modified_content
        
        # 重新构建完整的 overset 结构
        return f'\\overset{{{first_content}}}{{{modified_content}}}'
    
    # 首先找到所有 overset 的开始位置，然后手动处理每一个
    result = response
    
    # 使用简单的模式先找到所有 \overset{ 的位置
    overset_positions = []
    pos = 0
    while True:
        pos = result.find('\\overset{', pos)
        if pos == -1:
            break
        overset_positions.append(pos)
        pos += 1
    
    # 从后往前处理，避免位置偏移
    for start_pos in reversed(overset_positions):
        # 找到完整的 overset 结构
        overset_start = start_pos
        
        # 找到第一个花括号的结束位置
        first_brace_start = start_pos + len('\\overset')
        first_brace_end = find_matching_brace(result, first_brace_start)
        if first_brace_end == -1:
            continue
        
        # 找到第二个花括号的结束位置
        second_brace_start = first_brace_end + 1
        if second_brace_start >= len(result) or result[second_brace_start] != '{':
            continue
        
        second_brace_end = find_matching_brace(result, second_brace_start)
        if second_brace_end == -1:
            continue
        
        overset_end = second_brace_end + 1
        overset_text = result[overset_start:overset_end]
        
        # 创建一个假的 match 对象来兼容现有函数
        class FakeMatch:
            def __init__(self, text, start):
                self.text = text
                self.start_pos = start
            def group(self, n):
                return self.text
            def start(self):
                return self.start_pos
        
        fake_match = FakeMatch(overset_text, overset_start)
        processed_text = process_single_overset(fake_match)
        
        # 替换原文本
        result = result[:overset_start] + processed_text + result[overset_end:]
    
    return result


if __name__ == "__main__":
#     response = '''$\overset{huāng yě}{\tianzige{<ans><ref>荒</ref><box>[[45, 239, 85, 400]]</box></ans>}\tianzige{<ans><ref>野</ref><box>[[124, 241, 162, 397]]</box></ans>}}$ $\overset{zhèn yǔ}{\tianzige{<ans><ref>阵</ref><box>[[319, 241, 356, 391]]</box></ans>}\tianzige{<ans><ref>雨</ref><box>[[400, 239, 435, 378]]</box></ans>}}$ $\overset{kuáng huān}{\tianzige{<ans><ref>狂</ref><box>[[586, 241, 624, 386]]</box></ans>}\tianzige{<ans><ref>欢</ref><box>[[665, 245, 702, 372]]</box></ans>}}$ $\overset{shuāng bì}{\tianzige{<ans><ref>双</ref><box>[[855, 241, 895, 361]]</box></ans>}\tianzige{<ans><ref>臂</ref><box>[[926, 203, 967, 425]]</box></ans>}}$
# $\overset{gōng kè}{\tianzige{<ans><ref>功</ref><box>[[38, 760, 81, 917]]</box></ans>}\tianzige{<ans><ref>课</ref><box>[[122, 764, 170, 911]]</box></ans>}}$ $\overset{néng gòu}{\tianzige{<ans><ref>能</ref><box>[[319, 743, 357, 903]]</box></ans>}\tianzige{<ans><ref>够</ref><box>[[393, 743, 428, 888]]</box></ans>}}$ $\overset{hù xiāng}{\tianzige{<ans><ref>互</ref><box>[[590, 741, 620, 880]]</box></ans>}\tianzige{<ans><ref>相</ref><box>[[660, 737, 698, 880]]</box></ans>}}$ $\overset{zì rán}{\tianzige{<ans><ref>自</ref><box>[[860, 741, 884, 894]]</box></ans>}\tianzige{<ans><ref>然</ref><box>[[928, 733, 969, 903]]</box></ans>}}$'''

    response = r'1. “舞”字共$\underline{<ans><ref>14</ref><box>[[190, 60, 234, 500]]</box></ans>}$笔，第13笔是$\underline{<ans><ref>L</ref><box>[[452, 45, 491, 360]]</box></ans>}$。带有“舞”字的成语有$\underline{<ans><ref>张牙舞爪</ref><box>[[828, 0, 950, 410]]</box></ans>}$、$\underline{<ans><ref>闻鸡起舞</ref><box>[[28, 480, 193, 1000]]</box></ans>}$。'    
    
    res = merge_overset(response)
    print(res)