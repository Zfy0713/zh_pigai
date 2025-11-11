'''
基于搜题结果作比对批改
'''
import stat
from Levenshtein import distance
import csv
import os 
import json
import jieba
from tqdm import tqdm
import re
import ast, html
import numpy as np
import pandas as pd


from ..utils.utils import load_jsonL, flatten_nested_list

def eval_json(x):
    # x = "{'height': 129, 'is_print': False, 'text': '##二、用"✓"选择正确的读音.##', 'width': 694, 'x': 128, 'y': 34}\n{'height': 43, 'is_print': False, 'text': '##yóu##', 'width': 74, 'x': 432, 'y': 163, 'source': 'supp_detect'}\n{'height': 39, 'is_print': False, 'text': '##luè##', 'width': 57, 'x': 847, 'y': 153, 'source': 'supp_detect'}\n{'height': 36, 'is_print': False, 'text': '##kuò ##', 'width': 72, 'x': 1171, 'y': 144, 'source': 'supp_detect'}\n{'height': 45, 'is_print': False, 'text': '## gōu##', 'width': 80, 'x': 436, 'y': 240, 'source': 'supp_detect'}\n{'height': 39, 'is_print': False, 'text': '##xī ##', 'width': 38, 'x': 744, 'y': 237, 'source': 'supp_detect'}\n{'height': 46, 'is_print': False, 'text': '##zēng##', 'width': 101, 'x': 1317, 'y': 218, 'source': 'supp_detect'}"
    xx = html.unescape(x)  # 如果x是HTML转义的字符串，可以先进行解码
    # 将字符串按换行符拆分成多个字典字符串
    dict_strings = xx.split('\n')
    # 使用ast.literal_eval更安全地解析Python字面量
    result = []
    for dict_str in dict_strings:
        try:
            res = ast.literal_eval(dict_str)
        except Exception as e:
            print(f"Error parsing string: {dict_str}\nError: {e}")
            res = {}
        result.append(res)
    # result = [ast.literal_eval(dict_str) for dict_str in dict_strings]
    # 如果需要JSON字符串输出
    json_result = json.dumps(result, ensure_ascii=False, indent=2)
    # print(json_result)
    return result


class Pigai_Mhpp:
    def __init__(self):
        pass
    
    def get_topN(self, hand_text_info):
        '''
        获取手写体topN结果
        return: topN结果列表
        '''
        if 'topN' not in hand_text_info or len(hand_text_info['topN']) == 0:
            return [self.clean_text(hand_text_info['text'])]
        topN_res = [self.clean_text(hand_text_info['text'])] + [self.clean_text(j['text']) for j in hand_text_info['topN']['results'][:2] if j['prob_reg']>=0.5] ### topN取前两个置信度大于0.5
        topN_res += [self.clean_text(j['text']) for j in hand_text_info['topN']['results'][2:] if j['prob_reg'] >= 0.75] ### 后三个topN结果要求置信度大于0.75
        topN_res = list(dict.fromkeys(topN_res)) ###去重
        return topN_res

    def most_common(self, col):
        stat_res = {}
        for k in col: ### 统计批改结果和出现次数
            stat_res[k] = stat_res.get(k,0) + 1
        if 1 in stat_res:
            return 1
        elif 0 in stat_res:
            return 0
        else:
            return 9
    
    def remove_latex_commands(self, latex_str):
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
    
    def remove_punctuation_and_special_chars(self, text):
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

    def clean_text(self, text):
        text = self.remove_latex_commands(text)
        text = self.remove_punctuation_and_special_chars(text)
        text = text.replace('##', '')  ## 去掉##
        # 清理文本，去除标点符号只保留中文、英文、数字
        text = text.strip()
        text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
        ## 将对勾和叉号替换为对应的中文
        text = text.replace('√', '正确').replace('×', '错误')
        text = re.sub(r'\s+', '', text)  # 去除多余空格
        return text

    @staticmethod
    def to_Latin(char):
        TO_LATIN_MAP = {
            # 希腊字母 → 拉丁字母
            'Β': 'B',  # Greek Beta → B
            'Α': 'A',  # Greek Alpha → A
            'Ε': 'E',  # Greek Epsilon → E
            'Κ': 'K',  # Greek Kappa → K
            'Μ': 'M',  # Greek Mu → M
            'Ο': 'O',  # Greek Omicron → O
            'Ρ': 'P',  # Greek Rho → P
            'Τ': 'T',  # Greek Tau → T
            'Χ': 'X',  # Greek Chi → X
            'Υ': 'Y',  # Greek Upsilon → Y
        }
        return ''.join(TO_LATIN_MAP.get(c, c) for c in char)

    def is_choice(self, text):
        if len(text)>1:
            return False
        return text.upper() in ['A','B','C','D']

    def is_pinyin_str(self, text):
        # 拼音字母和带声调的拼音字母的范围
        pinyin_with_tones = "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ"
        pinyin_chars = f"a-zA-Z{pinyin_with_tones}"
        pattern = re.compile(rf'^[{pinyin_chars}]+$')
        return bool(pattern.match(text))
    
    def is_subsequence(self, sub, main):
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
    
    def longest_common_sublist(self, s1, s2, is_pinyin):
        '''
        找出匹配的手写体和答案
        input:
            s1, s2 - 分别对应手写体列表和标准答案列表
            is_pinyin - 是否是拼音题
        return: 
            lcs - 公共子列表序列
            common_pigai_res - 公共子列表序列对应的批改结果
            position1, position2 - 公共子列表序列在原列表中的位置
        '''
        m = len(s1)
        n = len(s2)
        
        L = [[0 for x in range(n+1)] for y in range(m+1)]
        for i in range(m+1):
            for j in range(n+1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif self._rag_mhpp_topN(s1[i-1], s2[j-1], is_pinyin) == 1:
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
            if (self._rag_mhpp_topN(s1[i-1], s2[j-1], is_pinyin) == 1):
                lcs.append(s2[j-1])
                common_pigai_res.append(self._rag_mhpp_topN(s1[i-1], s2[j-1], is_pinyin))
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

    def _rag_mhpp(self, hand_text, answer, is_pinyin): ### 字符串级别模糊匹配
        if '**' in answer: ### 教辅批改开放题答案回填形式 e.g. "三**土**干**大**上**下"
            answer = answer.split('**')
        else:
            answer = [answer]

        num_map = {
            '1': '一',
            '2': '二',
            '3': '三',
            '4': '四',
            '5': '五',
            '6': '六',
        }
        for ans in answer:
            if len(ans) == 1 and ans in num_map:
                answer.append(num_map[ans])
                
        
        def _mhpp(hand_text, answer, is_pinyin):
            hand_text = self.clean_text(hand_text)
            answer = self.clean_text(answer)

            hand_text = self.to_Latin(hand_text)
            answer = self.to_Latin(answer)

            ### 笔顺题
            # answer = answer.replace('|', '1')
            if hand_text == '1' and answer in ['|','丨']: ### 笔顺题特殊处理
                return 1
            # ㄴ
            if hand_text == 'L' and answer in ['ㄴ']: ### 笔顺题特殊处理
                return 1
                
            len_answer = len(answer)
            len_hand = len(hand_text)
            dis = distance(answer, hand_text)
            if self.is_choice(hand_text):
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
            # if is_pinyin and len_hand != len_answer:
            #     return 9
            if (self.is_pinyin_str(hand_text) or is_pinyin) and dis < 2:
                return 1
            
            return 9
        
        pg_res = any(_mhpp(hand_text, ans, is_pinyin) == 1 for ans in answer)
        if len(answer) == 1:
            return _mhpp(hand_text, answer[0], is_pinyin) ### 返回批改结果，1: 正确，0: 错误，9: 未批改
        else:
            return 1 if pg_res else 9 ### 返回批改结果，1: 正确，0: 错误，9: 未批改
        # if isinstance(answer, list): ### 教辅批改开放题答案回填形式 e.g. "三**土**干**大**上**下"
            

    
    def _rag_mhpp_topN(self, hand_text_info, answer: str, is_pinyin) -> int:
        '''
        手写体列表比对标准答案
        '''
        
        topN_res = self.get_topN(hand_text_info) ### 获取手写体topN结果

        temp_pigai_result = list(map(lambda x: self._rag_mhpp(x, answer, is_pinyin), topN_res))
        
        topN_pigai_res = {}
        for k in temp_pigai_result: ### 统计批改结果和出现次数
            topN_pigai_res[k] = topN_pigai_res.get(k,0) + 1
        if 1 in topN_pigai_res:
            return 1
        elif 0 in topN_pigai_res:
            return 0
        else:
            return 9
        
    def mhpp_run(self, hand_text_list, answer_list, is_pinyin):
        '''
        return: 手写体批改结果，1: 正确，0: 错误，9: 未批改
        '''
        
        answer_list = flatten_nested_list(answer_list) ### 将答案列表展开成一维列表
        if len(answer_list) == 0:
            return [9 for _ in range(len(hand_text_list))]
        
        for hand_text in hand_text_list:
            ### 去掉公式格式 ($$...$$)
            hand_text_prime = hand_text['text'].strip('##')
            if hand_text_prime.startswith('$$') and hand_text_prime.endswith('$$'):
                hand_text_list.remove(hand_text)
        pigai_result = [9 for _ in range(len(hand_text_list))] ## 每个手写体对应一个批改结果

        not_matched_answer = []
        ### 题库答案和手写体作答个数相同，按照位置批改
        if len(hand_text_list) == len(answer_list):
            for idx in range(len(hand_text_list)):
                pigai_result[idx] = self._rag_mhpp_topN(hand_text_list[idx], answer_list[idx], is_pinyin)
                if pigai_result[idx] == 9:
                    not_matched_answer.append(answer_list[idx])

        else:
            lcs, common_pigai_res, pos1, pos2 = self.longest_common_sublist(hand_text_list, answer_list, is_pinyin)
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
                
                topN_res = self.get_topN(hand_text_list[idx])
                for ans in not_matched_answer[:]:
                    # if any([self.clean_text(j) in ans for j in topN_res]):
                    # if any(self._rag_mhpp_topN(hand_text_list[idx], ans, is_pinyin) for j in topN_res):
                    if self._rag_mhpp_topN(hand_text_list[idx], ans, is_pinyin) == 1:
                        not_matched_answer.remove(ans)
                        pigai_result[idx] = 1
                        break
                # temp_pigai = [(clean_text(_hand_text_list[idx]) in ans) and (len(clean_text(_hand_text_list[idx]))>1) for ans in not_matched_answer]
                # if any(temp_pigai):
                #     pigai_result[idx] = 1
                # not_pigai_handtext.append(hand_text_list[idx])

        # assert len(pigai_result) == len(hand_text_list)
        return pigai_result

    def process(self, batch):
        '''
        batch: 待批改的题目json列表，每个sample为包含手写体、标准答案、topN信息,...：
        '''
        for item in batch:
            # url='0e54_5301ac1b-b53f-4c64-a0a1-0a1a26e087db.jpg'
            # if url not in item['url']:
                # continue
            ocr = item['ocr'].strip()
            title = item['title'].strip()
            hand_text_list = ast.literal_eval(item['hand_text_list']) ## 学生作答
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

            bukepi_keywords = ['部首','表格', '连一连','连线', '划去','笔顺','笔画',] ### 不可批题型
            pidui_keywords = ['句式','阅读', '大写字母', '小写字母'] ### 批对题型
            if any(keyword in title for keyword in bukepi_keywords+pidui_keywords) or any(keyword in ocr for keyword in bukepi_keywords+pidui_keywords):
                is_pidui = 1
            else:
                is_pidui = 0

            if is_moxie:
                ### 默写题都判对
                result = [1 for _ in range(len(hand_text_list))]

            elif is_pidui:
                result = [1 for _ in range(len(hand_text_list))]

            else:
                try:
                    rag_result_llama = ast.literal_eval(item['rag_result_llamaindex']) ## 搜题结果
                except Exception as e:
                    print(f"Error parsing rag_result_llama: {e}")
                    rag_result_llama = []
                    continue
                pigai_res = []
                for i in range(len(rag_result_llama)):
                    answer_list = ast.literal_eval(rag_result_llama[i]['answer']) if rag_result_llama else [] ## 搜题结果答案
                    # print(item['url'], rag_result_llama[i]['answer'])
                    pigai_result = self.mhpp_run(hand_text_list, answer_list, is_pinyin)
                    pigai_res.append(pigai_result)
                try:
                    result = np.apply_along_axis(self.most_common, axis=0, arr=pigai_res)
                    result = result.tolist()
                except Exception as e:
                    print("Error:", e)
                    print(pigai_res)
                    continue
            ### 储存最终题库批改结果
            matched_box = []
            for i, res in zip(hand_text_list, result):
                hand_text = i['text'].strip('##')
                if len(hand_text) >= 7:
                    res = 1
                
                # ### 题库未批改且手写体长度>=2，都批对
                # if res == 9 and len(hand_text) >= 2:
                #     res = 1
                
                matched_box.append({
                    "result": res,
                    "matched_boxes": [
                        {
                            "x": i['x'], "y": i['y'], "width": i['width'], "height": i['height'],'text': i['text']
                        }
                    ]
                })

            item['matched_box'] = matched_box
        return batch
    
class Pigai_Exact_match(Pigai_Mhpp):
    '''
    精确匹配, 重写 _rag_mhpp 和 process 方法 
    '''
    def __init__(self):
        super().__init__()

    def _rag_mhpp(self, hand_text, answer): ### 字符串级别模糊匹配
        '''
        严格匹配手写体和答案
        '''
        hand_text = self.clean_text(hand_text)
        answer = self.clean_text(answer)
        # len_answer = len(answer)
        len_hand = len(hand_text)
        dis = distance(answer, hand_text)
        if dis == 0:
            return 1
        if len_hand == 1: ### 单选/判断题
            if dis > 0:
                return 0
        else:
            return 9
    
    def longest_common_sublist(self, s1, s2):
        '''
        找出匹配的手写体和答案
        input:
            s1, s2 - 分别对应手写体列表和标准答案列表
        return: 
            lcs - 公共子列表序列
            common_pigai_res - 公共子列表序列对应的批改结果
            position1, position2 - 公共子列表序列在原列表中的位置
        '''
        m = len(s1)
        n = len(s2)
        
        L = [[0 for x in range(n+1)] for y in range(m+1)]
        for i in range(m+1):
            for j in range(n+1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif self._rag_mhpp(s1[i-1], s2[j-1]) == 1:
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
            if (self._rag_mhpp(s1[i-1], s2[j-1]) == 1):
                lcs.append(s2[j-1])
                common_pigai_res.append(self._rag_mhpp(s1[i-1], s2[j-1]))
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
    
    def mhpp_run(self, hand_text_list, answer_list):
        '''
        return: 手写体批改结果，1: 正确，0: 错误，9: 未批改
        '''
        pigai_result = [9 for _ in range(len(hand_text_list))] ## 每个手写体对应一个批改结果

        ### 如果answer_list是多维的，将answer_list展开成一维列表
        answer_list = flatten_nested_list(answer_list)
        if len(answer_list) == 0:
            return [9 for _ in range(len(hand_text_list))]
        
        not_matched_answer = []
        lcs, common_pigai_res, pos1, pos2 = self.longest_common_sublist(hand_text_list, answer_list)
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

        # for idx in range(len(pigai_result)):
        #     if pigai_result[idx] == 9: ### 未批改的手写体作答再次匹配

        #         topN_res = self.get_topN(hand_text_list[idx])
        #         for ans in not_matched_answer[:]:
        #             if any([self.clean_text(j) in ans for j in topN_res]):
        #                 not_matched_answer.remove(ans)
        #                 pigai_result[idx] = 1
        #                 break

        # assert len(pigai_result) == len(hand_text_list)
        return pigai_result

    def process(self, batch):
        for item in batch:
            # url='95598016-af1d-4728-b639-062cb369243d'
            # if url not in item['url']:
            #     continue
            # ocr = item['ocr'].strip()
            # hand_text_list_with_vertices = ast.literal_eval(item['hand_text_list_correct']) ## 学生作答
            hand_text_list_with_vertices = eval_json(item["hand_text_list_correct"]) ## 人工矫正格式解析
            if len(hand_text_list_with_vertices) == 0:
                continue
            ocr = item['校正OCR'].strip()
            pattern = r'##(.*?)##'
            # hand_text_list = re.findall(pattern, ocr)
            hand_text_list = [item['text'].replace("##", "") for item in hand_text_list_with_vertices]

            answer_refine = item['题库答案校正']
            if len(answer_refine) > 0:
                answer_list = ast.literal_eval(answer_refine)
            else:
                rag_position = item['原题搜到位置']
                digits = re.findall(r'\d', rag_position)         
                answer_list = [ast.literal_eval(item[f'rag_answer{str(int(i))}']) for i in digits]
                answer_list = flatten_nested_list(answer_list) ### 将答案列表展开成一维列表
            
            pigai_result = self.mhpp_run(hand_text_list, answer_list)
            matched_box = []
            for i, res in zip(hand_text_list_with_vertices, pigai_result):
                matched_box.append({
                    "result": res,
                    "matched_boxes": [
                        {
                            "x": i['x'], "y": i['y'], "width": i['width'], "height": i['height'],'text': i['text']
                        }
                    ]
                })

            item['matched_box'] = matched_box
        return batch

class Pigai_Mhpp_new(Pigai_Mhpp):
    '''
    新版批改，支持模糊匹配和精确匹配
    '''
    def __init__(self):
        super().__init__()

    def answer_extract(self, question_str, answer_str, title_str=""):
        '''答案抽取接口 + 答案分割 --> 分割后答案列表'''
        answer_extract_res = req_answer_extract(question_str, answer_str)
        if answer_extract_res is None:
            answer_extract_res = answer_str
            
        pattern = r'^[0-9零一二三四五六七八九]+'
        rag_answer = re.sub(pattern, '', answer_extract_res)
        def remove_spaces_inside_dollar(s): ### 去掉$$...$$中的空格
            return re.sub(r'\$\$(.*?)\$\$', lambda match: '$$' + re.sub(r'\s+', '', match.group(1)) + '$$', s)
        
        rag_answer = remove_spaces_inside_dollar(rag_answer)
        is_xuhao = False
        if '序号' in question_str or '序号' in title_str:
            is_xuhao = True
        if is_xuhao: ## 序号题中带圈的数字 “①②③④⑤⑥……” 不作为题号分割
            pattern = r'【\d+】|[\d+]|（\d+）|\d+\.|\(\d+\)|\d+|；|;|、|，|。|！|？|\n| |【小题\d+】|\[小题\d+\]|【示例】|[示例]|<示例>|\$\$' ### 按照数字、括号数字、分号、换行符分割题库答案
        else:
            pattern = r'【\d+】|[\d+]|（\d+）|\d+\.|\(\d+\)|\d+|；|;|、|，|。|！|？|\n| |【小题\d+】|\[小题\d+\]|【示例】|[示例]|<示例>|[\u2460-\u2473]|\$\$' ### 按照数字、括号数字、分号、换行符分割题库答案
        answer_list = re.split(pattern, rag_answer)

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
        
        answer_list = [remove_latex_commands(i) for i in answer_list if i.strip() != ''] ### 去掉空字符串和latex命令
        return answer_list
        


    def process(self, batch, save_dir=None):

        # save_dir='/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/jiaofu/db_data/3books-0808/54ye-rag-mhpp/temp_save'
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        outputs = []
        for idx, item in enumerate(tqdm(batch)):
            url = item['url']
            img_name = os.path.split(url)[-1]

            if save_dir:
                save_path = os.path.join(save_dir, f"{img_name}_{idx}.json")
                if os.path.exists(save_path):
                    print(f"File {save_path} already exists, skipping...")
                    temp_json = load_json(save_path)
                    outputs.append(temp_json)
                    continue

            # url='95598016-af1d-4728-b639-062cb369243d'
            # if url not in item['url']:
            #     continue
            
            ocr = item['ocr'].strip()
            hand_text_list = ast.literal_eval(item["hand_text_list"])
            if len(hand_text_list) == 0:
                continue
            result = [9 for _ in range(len(hand_text_list))]


            if ocr.startswith(item['title']):
                ocr = ocr[len(item['title']):].strip()

            pinyin_keywords = ['看拼音','读拼音', '读音']
            is_pinyin = any([keyword in item['title'] for keyword in pinyin_keywords])
            item['is_pinyin'] = is_pinyin
            moxie_keywords = ['默写', '课文背诵', '日积月累', '古诗词积累', '诗词积累', '名句积累', '古诗积累', '必背', '经典诗词']
            if any(keyword in item['title'] for keyword in moxie_keywords) and not is_pinyin:
                is_moxie = 1
            else:
                is_moxie = 0
            item['is_moxie'] = is_moxie

            bukepi_keywords = ['部首','表格', '连一连','连线', '划去','笔顺','笔画',] ### 不可批题型
            pidui_keywords = ['句式','阅读', '大写字母', '小写字母'] ### 批对题型
            if any(keyword in item['title'] for keyword in bukepi_keywords+pidui_keywords) or any(keyword in ocr for keyword in bukepi_keywords+pidui_keywords):
                is_pidui = 1
            else:
                is_pidui = 0
            item['is_pidui'] = is_pidui

            if is_moxie:
                ### 默写题都判对
                result = [1 for _ in range(len(hand_text_list))]

            elif is_pidui:
                result = [1 for _ in range(len(hand_text_list))]

            else:
                rag_result = [item[f'rag{i+1}'] for i in range(3) if f'rag{i+1}' in item and item[f'rag{i+1}']]
                pigai_res = []
                for rag in rag_result:
                    rag = ast.literal_eval(rag) if isinstance(rag, str) else rag
                    ques = rag['question']
                    answer = rag['metadata']['answer']
                    if isinstance(answer, list):
                        answer_list = answer
                    # elif isinstance(ast.literal_eval(answer), list):
                    #     answer_list = ast.literal_eval(answer)
                    else:
                        print('answer extracting...')
                        answer_list = self.answer_extract(ques, answer, title_str = "")
                    pigai_result = self.mhpp_run(hand_text_list, answer_list, is_pinyin)
                    pigai_res.append(pigai_result)
                    try:
                        result = np.apply_along_axis(self.most_common, axis=0, arr=pigai_res)
                        result = result.tolist()
                    except Exception as e:
                        print("Error:", e)
                        print(pigai_res)
                        
                        continue
            # item['answer_list'] = answer_list
            ### 储存最终题库批改结果
            matched_box = []
            for i, res in zip(hand_text_list, result):
                hand_text = i['text'].strip('##')
                if len(hand_text) >= 7:
                    res = 1
                
                # ### 题库未批改且手写体长度>=2，都批对
                # if res == 9 and len(hand_text) >= 2:
                #     res = 1
                
                matched_box.append({
                    "result": res,
                    "matched_boxes": [
                        {
                            "x": i['x'], "y": i['y'], "width": i['width'], "height": i['height'],'text': i['text']
                        }
                    ]
                })
            item['matched_box'] = matched_box

            outputs.append(item)
            if save_dir:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(item, f, ensure_ascii=False, indent=2)
            
        return outputs

class Pigai_Mhpp_SmartMatch(Pigai_Mhpp):
    '''
    支持整页搜到结果, 重写 process 方法：去除答案抽取过程，只用top1搜到结果
    mhpp 方法：去掉未批改的手写体作答再次匹配的过程
    '''
    def __init__(self):
        super().__init__()

    def mhpp_run(self, hand_text_list, answer_list, is_pinyin):
        '''
        return: 手写体批改结果，1: 正确，0: 错误，9: 未批改
        '''

        answer_list = flatten_nested_list(answer_list) ### 将答案列表展开成一维列表

        if len(answer_list) == 0:
            return [9 for _ in range(len(hand_text_list))]
        
        pigai_result = [9 for _ in range(len(hand_text_list))] ## 每个手写体对应一个批改结果

        not_matched_answer = []
        ### 题库答案和手写体作答个数相同，按照位置批改
        if len(hand_text_list) == len(answer_list):
            for idx in range(len(hand_text_list)):
                pigai_result[idx] = self._rag_mhpp_topN(hand_text_list[idx], answer_list[idx], is_pinyin)
                if pigai_result[idx] == 9:
                    not_matched_answer.append(answer_list[idx])

        else:
            lcs, common_pigai_res, pos1, pos2 = self.longest_common_sublist(hand_text_list, answer_list, is_pinyin)
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
                
                topN_res = self.get_topN(hand_text_list[idx])
                for ans in not_matched_answer[:]:
                    # if any([self.clean_text(j) in ans for j in topN_res]):
                    # if any(self._rag_mhpp_topN(hand_text_list[idx], ans, is_pinyin) for j in topN_res):
                    if self._rag_mhpp_topN(hand_text_list[idx], ans, is_pinyin) == 1:
                        not_matched_answer.remove(ans)
                        pigai_result[idx] = 1
                        break
                # temp_pigai = [(clean_text(_hand_text_list[idx]) in ans) and (len(clean_text(_hand_text_list[idx]))>1) for ans in not_matched_answer]
                # if any(temp_pigai):
                #     pigai_result[idx] = 1
                # not_pigai_handtext.append(hand_text_list[idx])

        # assert len(pigai_result) == len(hand_text_list)
        return pigai_result

    def process(self, item):

        url = item['img_url']
        img_name = os.path.split(url)[-1]
        hand_text_list = ast.literal_eval(item["hand_text_list"]) if isinstance(item["hand_text_list"], str) else item["hand_text_list"]
        if len(hand_text_list) == 0:
            return
        ocr = item['text_vl']
        title = item['title']
        if ocr.startswith(item['title']):
            ocr = ocr[len(item['title']):].strip()

        pinyin_keywords = ['看拼音','读拼音', '读音']
        is_pinyin = any([keyword in item['title'] for keyword in pinyin_keywords])
        item['is_pinyin'] = is_pinyin
        moxie_keywords = ['默写', '课文背诵', '日积月累', '古诗词积累', '诗词积累', '名句积累', '古诗积累', '必背', '经典诗词']
        if any(keyword in item['title'] for keyword in moxie_keywords) and not is_pinyin:
            is_moxie = 1
        else:
            is_moxie = 0
        item['is_moxie'] = is_moxie

        bukepi_keywords = ['部首','表格', '连一连','连线', '划去','笔顺','笔画',] ### 不可批题型
        pidui_keywords = [] ### 批对题型
        if any(keyword in item['title'] for keyword in bukepi_keywords + pidui_keywords) or any(keyword in ocr for keyword in bukepi_keywords+pidui_keywords):
            is_pidui = 1
        else:
            is_pidui = 0


        rag_result = item['find_within_page']
        answer_list = rag_result['ht_answer_text']
        type_list = rag_result['type_list']
        item['answer_list'] = answer_list

        if isinstance(answer_list, str):
            answer_list = ast.literal_eval(answer_list)
        if len(answer_list) == 0:
            return
        

        pigai_result = self.mhpp_run(hand_text_list, answer_list, is_pinyin)
        item['segmented_answers'] = []
        ### 答案分词
        answer_list = flatten_nested_list(answer_list)
        chinese_pattern = re.compile(r'^[\u4e00-\u9fff]$')
        def is_chinese_char(ch):
            return bool(chinese_pattern.match(ch))

        if all(len(ans)==1 for ans in answer_list if is_chinese_char(ans)): ### 全是单个汉字答案，进行分词处理
            segmented_answers = self.cut_chinese_segments(answer_list)
            item['segmented_answers'] = segmented_answers
            pigai_result_segmented = self.mhpp_run(hand_text_list, segmented_answers, is_pinyin)
        
            ## 综合两种答案分割方式的批改结果
            final_pigai_result = []
            for res1, res2 in zip(pigai_result, pigai_result_segmented):
                if res1 == 1 or res2 == 1:  
                    final_pigai_result.append(1)
                elif res1 == 0 and res2 == 0:
                    final_pigai_result.append(0)
                else:
                    final_pigai_result.append(9)
            pigai_result = final_pigai_result

        # if is_moxie:
        #     result = [1 for i in range(len(hand_text_list)) if pigai_result[i] == 9]

        if is_pidui:
            result = [1 for i in range(len(hand_text_list)) if pigai_result[i] == 9]
        else:
            result = pigai_result
        
        # if ocr in ocr_test:
        #     print(result)
        
        ### 储存最终题库批改结果
        matched_box = []
        for i, res in zip(hand_text_list, result):
            x1, y1 = i['x'], i['y']
            width = i['width']
            height = i['height']
            x2, y2 = x1 + width, y1 + height
            hand_text = i['text'].strip('##')
            matched_box.append({
                "x": x1 + (x2-x1)//2,
                "y": y1 + (y2-y1)//2,
                "result": res,
                "box": [x1, y1, x2, y2] 
            })

        item['matched_box'] = matched_box
            
        return {
            "img_url": item['img_url'],
            "matched_box": matched_box
        }
        
    def cut_chinese_segments(self, char_list):
        result = []
        buffer = []
        chinese_pattern = re.compile(r'^[\u4e00-\u9fff]$')  # 单个常用汉字
        for ch in char_list:
            if chinese_pattern.match(ch):
                buffer.append(ch)
            else:
                # 遇到非中文，先处理缓存中的中文
                if buffer:
                    result.extend(jieba.cut(''.join(buffer)))
                    buffer = []        
                result.append(ch)
        # 处理尾部的中文缓存
        if buffer:
            result.extend(jieba.cut(''.join(buffer)))
        return list(result)

def main(args):
    from collections import defaultdict

    input_file = args.input_file
    save_path = args.save_path
    data = load_jsonL(input_file)

    Pigai_obj = Pigai_Mhpp_SmartMatch()
    # outputs = Pigai_obj.process(data)
    draw_output = defaultdict(list)
    for item in data:
        img_url = item['img_url']
        output = Pigai_obj.process(item)
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
    parser.add_argument("--input_file", type=str, required=True, help="")
    parser.add_argument("--save_path", type=str, required=True, help="")
    args = parser.parse_args()
    main(args)