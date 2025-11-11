import ast
import requests
import json, os, csv, sys, time, datetime
import base64
from tqdm import tqdm
from pathlib import Path
import warnings
import threading
from ..utils.utils import load_json, load_jsonL, save_json, save_jsonL, load_csv_2_dict, save_json_2_csv

import logging
logging.basicConfig(level=logging.INFO)

PROMPT_TEMPLATE="""你是一位经验丰富的中国K12语文老师，你现在的任务是批改学生作答的语文题目。

【任务内容】
输入格式：<题目>为OCR识别到的试卷题目，题目中的学生答案以“##”符号分隔。需要提取出全部的学生答案，每个学生答案都要返回一个批改结果。你需要生成一个严格遵循格式的JSON格式<批改结果>，不进行任何自定义修改。

【输出规范】
1.输出的json格式包含“data”字段，该字段是一个包含所有题目的列表。列表中的每个json字典对应一个题目的批改结果。
2.每道题必须包含两个字段，分别是：“response”、“type”。
3.“response”字段中包含对这道题的全部小空的批改结果。每个小空都有以下三个字段：“hand_text”字段、“answer”字段和“result”字段。
    4.1.“hand_text”字段为学生答案。该字段的值是从“##”中提取出来的，例如，##A)## 提取到的内容为A，则“hand_text”字段的值为A。忽略学生答案中的特殊符号。
    4.2.“answer”字段为每一空的正确答案，你需要根据上下文理解题意，并做出正确的解答。
    4.3.“result”字段为对学生答案的批改结果。批改结果必须清晰明确：“正确”、“错误”或“”，避免模糊表述如“部分正确”。若“hand_text”字段与“answer”字段的结果完全一致或基本一致，则批改结果为“正确”，否则为“错误”。需要注意的是，填空题中，可以包容学生答案中出现标点符号、大小写、特殊字符等问题。若“hand_text”字段为空，表示学生没有作答，则result结果为空，无需给出批改结果。
4.“type”字段为题目的类型，你根据已知并自己判断类型，包括选择题、填空题、阅读题、拼音写字题、句子改写题等等。
5.由于是图片扫描得到的文本，你必须容忍学生的拼写错误和书写错误。
6.你必须严谨并准确地批改学生的作答，如果批改错误，你将受到惩罚。学生的试卷可能出现不完整的情况，不要自行补充。

【示例】
用户输入：我发现带“礻”的字多与 ##神## 有关,带“衤”的字多与##衣服##有关。\n8. “三顾茅庐”出自哪个故事?(##B##)A.刘备拜见诸葛亮 B.诸葛亮拜见刘备 C.诸葛亮拜见曹操
【输出】
{{
    "data": [
        {{
            "response": [
                {{
                    "hand_text": "神",
                    "answer": "神",
                    "result": "正确"
                }},
                {{
                    "hand_text": "衣服",
                    "answer": "衣服",
                    "result": "正确"
                }}
            ],
            "type": "填空"
        }},
        {{
            "response": [
                {{
                    "hand_text": "B",
                    "answer": "A",
                    "result": "错误"
                }}
            ],
            "type": "选择"
        }}
    ]
}}
【用户输入】
{question}
【输出】"""

class Qwen_api:
    def __init__(self, url, MODEL_NAME, enable_thinking=False):
        self.url = url
        self.MODEL_NAME = MODEL_NAME
        self.enable_thinking = enable_thinking

    def generate(self, prompt):
        headers = {
            "Authorization": "Bearer sk-e9925549b2fd42b4a937d4009e9a03f7",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "top_p": 0.7,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "enable_thinking": self.enable_thinking,
            "stream": True if self.enable_thinking else False,
            # "extra_body": {"extra_body.model": MODEL_NAME}
        }
        if self.enable_thinking:
            try:
                response = requests.post(self.url, headers=headers, json=data, stream=True)
                # 检查响应状态码
                response.raise_for_status()

                # 处理流式数据
                print("Streaming response:")
                res_content = ""
                res_reason = ""
                for chunk in response.iter_lines(decode_unicode=True):
                    if chunk:  # 避免处理空行
                        try:
                            res = json.loads(chunk[5:])
                            if res['choices'][0]['finish_reason'] == 'stop':
                                break
                            if res['choices'][0]['delta']['content']:
                                res_content += res['choices'][0]['delta']['content']
                            if res['choices'][0]['delta']['reasoning_content']:
                                res_reason += res['choices'][0]['delta']['reasoning_content']
                            # res = {
                            #     'content': res_content,
                            #     'reasoning_content': res_reason,
                            # }
                        except json.JSONDecodeError as e:
                            res = {
                                "content": f"Error decoding JSON: {e}"
                            }
                        # print(res)  # 逐行打印流数据
                res = {
                    'content': res_content,
                    'reasoning_content': res_reason,
                }
                
            except requests.exceptions.RequestException as e:
                res = {
                    "content": f"Error during inference: {e}"
                }
                # print(f"An error occurred: {e}")

        else: 
            try:
                response = requests.post(self.url, headers=headers, json=data)
                if response.status_code == 200:
                    res = response.json()
                    # print(res)
                    res = {
                        "content": res['choices'][0]['message']['content']
                    }
                else:
                    res = {
                        "content": f"Error: {response.status_code}, {response.text}"
                    }
            except Exception as e:
                res = {
                    "content": f"Error during inference: {e}"
                }
        return res

    def single_run(self, item):
        problem = item['text_vl'].replace('"', '')
        title = item['title'] if 'title' in item else ""
        problem = title + problem
        input_ = PROMPT_TEMPLATE.format(question=problem)
        res = self.generate(input_)
        item['model_output'] = res['content']

    def process_items(self, items):
        for item in items:
            if 'model_output' in item and not item['model_output'].startswith('Error'):
                continue
            self.single_run(item)
            time.sleep(2)

    def run(self, input_path, output_path, NUM_THREADS=5):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = load_jsonL(input_path)

        if NUM_THREADS > 1:

            chunk_size = (len(data) + NUM_THREADS - 1)// NUM_THREADS
            threads = []
            for i in range(0, len(data), chunk_size):
                items = data[i: min(i + chunk_size, len(data))]
                t = threading.Thread(target=self.process_items, args=(items,))
                threads.append(t)
                t.start()
            for t in tqdm(threads):
                t.join()

        else:
            self.process_items(data)
        save_json_2_csv(data, output_path)
        # save_jsonL(output_path, data)


if __name__ == "__main__":
    # input_path='/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/0418data-new/ocr_supp_chaiti_rag.csv'
    # output_path='/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/0418data-new/qwen-plus-2025-04-28-api/model_output.csv'

    import argparse
    parser = argparse.ArgumentParser()
  
    # Add arguments for input and output file paths
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input jsonl file.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the output jsonl file.")
    parser.add_argument('--MODEL_NAME', type=str, default='qwen2.5-72b-instruct', help="Model name.")
    parser.add_argument('--NUM_THREADS', type=int, default=5, help="Number of threads to use for processing.")
    # parser.add_argument('--enable_thinking', action='store_true', help="Enable thinking mode.")
    parser.add_argument('--api_url', type=str, default='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions', help="API URL.")

    # Parse the arguments
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    # os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # MODEL_NAME="qwen2.5-72b-instruct"
    # MODEL_NAME='qwen-plus-2025-04-28' ### qwen3
    # MODEL_NAME='qwen-max'
    
    qwen_api = Qwen_api(url=args.api_url, MODEL_NAME=args.MODEL_NAME, enable_thinking=False)
    qwen_api.run(input_path, output_path, NUM_THREADS=args.NUM_THREADS)

    # URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    # MODEL_NAME="qwen2.5-72b-instruct"

    # qwen_api = Qwen_api(url=URL, MODEL_NAME=MODEL_NAME, enable_thinking=False)
    # prompt = "请给出一段关于人工智能的简短介绍。"
    # result = qwen_api.generate(prompt)
    # print(result)