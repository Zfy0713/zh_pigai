import json, csv
import argparse
import numpy as np
import re
import requests
import time
from pathlib import Path
# from utils.utils import load_csv_2_dict, load_jsonL, save_jsonL, load_json, save_json
from ..utils.utils import load_json, load_jsonL, save_json, save_jsonL, load_csv_2_dict


import os
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import AutoTokenizer, AutoProcessor
from vllm import LLM, SamplingParams

QWEN_IM_START = '<|im_start|>'
QWEN_IM_START_ID = 151644
QWEN_IM_END = '<|im_end|>'
QWEN_IM_END_ID = 151645
QWEN_SUCCESS = 'Y'
QWEN_SUCCESS_ID = 56
QWEN_FAIL = 'N'
QWEN_FAIL_ID = 45

LLAMA_START = '<|begin_of_text|>'
LLAMA_START_ID = 128000
LLAMA_END = '<|end_of_text|>'
LLAMA_END_ID = 128001
LLAMA_EOT_ID = 128009

def save_csv(save_path, json_data):
    headline = list(json_data[0].keys())
    with open(save_path, 'w', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headline)
        for line in json_data:
            if "##" not in str(line['ocr']):
                line['extract'] = ""
            content = list(line.values())
            
            # content = line['response']
            w.writerow(content)

def remove_hash_enclosed(text):
    # 使用正则表达式找到并替换所有被“##”包裹的文本，包括“##”
    result = re.sub(r'##.*?##', '', text)
    return result

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0, help="")
    parser.add_argument(
        "--dtype",
        type=str,
        help="float16 or int8",
        choices=["int8", "float16"],
        default="float16",
    )
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--data_path", type=str, help="", required=True)
    parser.add_argument("--max_gen_length", type=int, help="", default=1024)
    parser.add_argument("--num_return_sequences", type=int, default=1, help="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft_model_path", type=str, default=None)
    parser.add_argument("--task_type", type=str, default="correct")
    parser.add_argument("--gen_prob", action='store_true', help="whether to generate probability")
    parser.add_argument("--model_type", type=str, choices=['qwen','deepseek','llama'], default="qwen")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enable_thinking", action='store_true', help="Whether enable thinking mode (Qwen3 only).")

    parser.add_argument("--save_path", type=str, default=None, help="")

    return parser.parse_args()


class LLMInference:
    def __init__(self, args):
        self.args = args
        # self.dataset = load_jsonL(args.data_path)
        if args.data_path.endswith('jsonl'):
            self.dataset = load_jsonL(args.data_path)
        elif args.data_path.endswith('csv'):
            self.dataset = load_csv_2_dict(args.data_path)
        else:
            self.dataset = load_json(args.data_path)
        
        sub_inputs = list(map(self.combine_prompt, self.dataset))
        self.model = LLM(model=args.sft_model_path, tensor_parallel_size=args.num_gpus, seed=args.seed,
                     trust_remote_code=True, gpu_memory_utilization=args.gpu_memory_utilization, 
                     enforce_eager=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
        # if args.model_type == 'deepseek':
        #     self.tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path,
        #                                                trust_remote_code=True)
        
        self.gen_kwargs = {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "temperature": args.temperature,
            "max_tokens": args.max_gen_length,
            "n": args.num_return_sequences,
            "presence_penalty": args.presence_penalty,
            "frequency_penalty": args.frequency_penalty,
            "repetition_penalty": args.repetition_penalty,
            "skip_special_tokens": False,
        }

        if args.model_type == 'qwen':
            self.gen_kwargs.update({
                "stop": [QWEN_IM_END, QWEN_IM_START],
                "stop_token_ids": [QWEN_IM_END_ID, QWEN_IM_START_ID]
            })
        elif args.model_type == 'llama':
            self.gen_kwargs.update({
                "stop_token_ids": [LLAMA_EOT_ID]
            })
        else:
            raise ValueError
         
        if args.gen_prob:
            self.gen_kwargs.update({
                "logprobs": 5
            })

        self.sampling_params = SamplingParams(**self.gen_kwargs)
    
    def generate(self, batch):
        output_text = ['' for i in range(len(batch))]
        if 'Qwen3' in self.args.sft_model_path:
            _batch = [
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ] for prompt in batch
            ]
            text = self.tokenizer.apply_chat_template(
                _batch,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking= self.args.enable_thinking # Switches between thinking and non-thinking modes. Default is True.
            )
            if self.args.enable_thinking:
                print(f'========{os.path.split(self.args.sft_model_path)[-1]} thinking mode!!!!========')
                self.gen_kwargs.update({
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0.0,
                })
            else: 
                # non-thinking mode
                print(f'========{os.path.split(self.args.sft_model_path)[-1]} non-thinking mode!!!!========')
                self.gen_kwargs.update({
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "min_p": 0.0,
                })
            self.sampling_params = SamplingParams(**self.gen_kwargs)
            outputs = self.model.generate(text, sampling_params=self.sampling_params)

        else:
            outputs = self.model.generate(batch, self.sampling_params)
        
        assert len(batch) == len(outputs), f"batch length: {len(batch)} not equal to output length {len(output)}!"
        for output in outputs:
            # print(output.outputs)
            text = output.outputs[0].text
            output_text[int(output.request_id)] = text
        return output_text

    def chat(self, batch):
        '''
        chat_template for (non-)thinking mode in Qwen3
        '''
        output_text = ['' for i in range(len(batch))]
        ### Qwen3 only
        if self.args.enable_thinking:
            print('non-thinking mode does not work!!!!')
            self.gen_kwargs.update({
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
            })
        else: 
            # non-thinking mode
            print('non-thinking mode works!!!!')
            self.gen_kwargs.update({
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
            })

        self.sampling_params = SamplingParams(**self.gen_kwargs)

        messages = [
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ] for prompt in batch
        ]
        outputs = self.model.chat(
            messages,
            sampling_params=self.sampling_params,
            chat_template_kwargs={"enable_thinking": self.args.enable_thinking},  # or True
        )
        for output in outputs:
            text = output.outputs[0].text
            output_text[int(output.request_id)] = text

        return output_text

    def combine_prompt(self, sample):

        if self.args.task_type == 'pigai_yuwen':

            # problem = sample['ocr'].replace('"','')
            problem = sample['text_vl'].replace('"','')
            title = sample['title'] if 'title' in sample else ""

            problem = title + problem
            # is_ocr = sample['OCR是否正确']
            # if is_ocr == '错误':
            #     problem = sample['校正OCR']
            # else:
            #     problem = sample['ocr']
            prompt_template = """你是一位经验丰富的中国K12语文老师，你现在的任务是批改学生作答的语文题目。

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
            input_ = prompt_template.format(question=problem)
            system_prompt = ""
        
        elif self.args.task_type == 'pigai_yuwen_new':
            problem = sample['ocr'].replace('"','')
            PROMPT_TEMPALTE_new = """你是一位经验丰富的中国K12语文老师，你现在的任务是批改学生作答的语文题目。
【任务内容】
用户输入包含K12语文题目，用户输入相应的OCR识别到的题目内容，题目中的学生答案以"##"符号分隔。需要提取出全部的学生答案，每个学生答案都要返回一个批改结果。你需要生成一个严格遵循格式的JSON格式<批改结果>，不进行任何自定义修改。
【输出规范】
1.输出的json格式包含"data"字段，该字段是一个包含所有题目的列表。列表中的每个json字典对应一个题目的批改结果。
2.每道题必须包含两个字段，分别是："response"、"type"。
3."response"字段中包含对这道题的全部学生答案的批改结果。每个小空都有以下三个字段："hand_text"字段、"answer"字段和"result"字段。
    3.1."hand_text"字段为学生答案。该字段的值必须严格从用户输入的文本中提取出来，提取规则是提取"##"中间部分的内容，例如，##A)## 提取到的内容为A，则"hand_text"字段的值为A。忽略学生答案中的特殊符号。"hand_text"字段的值必须是用户输入文本中"##"中间部分的内容，不能随意修改。"hand_text"的值不能是空字符串。
    3.2."answer"字段为每一空的正确答案，你需要根据上下文理解题意，并做出正确的解答。
    3.3."result"字段为对学生答案的批改结果。批改结果必须清晰明确："正确"、"错误"或""，避免模糊表述如"部分正确"。若"hand_text"字段与"answer"字段的结果完全一致或基本一致，则批改结果为"正确"，否则为"错误"。需要注意的是，填空题中，可以包容学生答案中出现标点符号、大小写、特殊字符等问题。若"hand_text"字段为空，表示学生没有作答，则result结果为空，无需给出批改结果。
4."type"字段为题目的类型，你根据已知并自己判断类型，包括选择题、填空题、阅读题、拼音写字题、句子改写题等等。
5.由于是图片扫描得到的文本，你必须容忍学生的拼写错误和书写错误。
6.你必须严谨并准确地批改学生的作答，如果批改错误，你将受到惩罚。学生的试卷可能出现不完整的情况，不要自行补充。
7.只批改带有学生手写体的空，没有学生手写体的空不需要输出批改结果。
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
【示例】
用户输入：一、解释加点的词语。 1. 陈康肃公善射( ##善于## ) 2. 公亦以此自矜( ##夸耀## ) 3. 有卖油翁释担而立( ##放下## ) 4. 但微颔之( ##点头## )( ##而已## ) 5. 吾射不亦精乎( ) 6. 尔安敢轻吾射( )( ) 7. 乃取一葫芦置于地( 8. 以钱覆其口( ) 9. 徐以杓酌油沥之( ) 10. 康肃笑而造之( ) 11. 射 (1)尝射于家圃( ) (2)尔安敢轻吾射( ) 12. 之 (1)但微颔之( ) (2)以我酌油知之( ) (3)以杓酌油沥之( ) (4)笑而遗之( ) 13. 其 (1)见其发矢十中八九( ) (2)以钱覆其口( ) 14. 以 (1)以我酌油知之( ) (2)以钱覆其口( ) 15. 而 (1)释担而立( ) (2)自钱孔入,而钱不湿( ) (3)康肃笑而遣之( ) 二、用现代汉语翻译句子。
【输出】
{{
    "data": [
        {{
            "response": [
                {{
                    "hand_text": "善于",
                    "answer": "善于",
                    "result": "正确"
                }},
                {{
                    "hand_text": "夸耀",
                    "answer": "夸耀",
                    "result": "正确"
                }},
                {{
                    "hand_text": "放下",
                    "answer": "放下",
                    "result": "正确"
                }},
                {{
                    "hand_text": "点头",
                    "answer": "只",
                    "result": "错误"
                }},
                {{
                    "hand_text": "而已",
                    "answer": "点头",
                    "result": "错误"
                }}
            ],
            "type": "解释词语"
        }}
    ]
}}
【用户输入】
{question}
【模型输出】
"""
            input_ = PROMPT_TEMPALTE_new.format(question=problem)
            system_prompt = ""
        
        elif self.args.task_type == 'jieti_yuwen':
            problem = sample['ocr']
            prompt_template = """你是一位经验丰富的中国K12语文老师，你现在的任务是解答语文题目。
            【任务内容】输入为OCR识别到的试卷题目，你需要根据题目内容进行解答。
            【输出规范】
            1.请直接输出答案，不要包含任何其他内容（解析、解释等）。如果有多个空，用序号区分。
            2.用户输入学生手写体以“##”符号分隔，例如“##A##”提取到的内容为“A”。请你只作答带有学生手写体的空，没有学生手写体的空不需要输出答案。
            【示例1】
            用户输入：我发现带“礻”的字多与（  ）有关,带“衤”的字多与（  ）有关。\n8. “三顾茅庐”出自哪个故事?(  )A.刘备拜见诸葛亮 B.诸葛亮拜见刘备 C.诸葛亮拜见曹操
            【输出】
            1. 神
            2. 衣服
            3. B

            【示例2】
            用户输入：一、解释加点的词语。 1. 陈康肃公善射( ##善于## ) 2. 公亦以此自矜( ##夸耀## ) 3. 有卖油翁释担而立( ##放下## ) 4. 但微颔之( ##点头## )( ##而已## ) 5. 吾射不亦精乎( ) 6. 尔安敢轻吾射( )( ) 7. 乃取一葫芦置于地( 8. 以钱覆其口( ) 9. 徐以杓酌油沥之( ) 10. 康肃笑而造之( ) 11. 射 (1)尝射于家圃( ) (2)尔安敢轻吾射( ) 12. 之 (1)但微颔之( ) (2)以我酌油知之( ) (3)以杓酌油沥之( ) (4)笑而遗之( ) 13. 其 (1)见其发矢十中八九( ) (2)以钱覆其口( ) 14. 以 (1)以我酌油知之( ) (2)以钱覆其口( ) 15. 而 (1)释担而立( ) (2)自钱孔入,而钱不湿( ) (3)康肃笑而遣之( ) 二、用现代汉语翻译句子。
            【输出】
            1. 善于
            2. 夸耀
            3. 放下
            4. 只
            5. 点头
            【用户输入】
            {question}
            【输出】"""
            input_ = prompt_template.format(question=problem)
            system_prompt = ""
        
        else:
            raise ValueError

        if self.args.model_type == "qwen":
            prompt = f"{QWEN_IM_START}system\n{system_prompt}{QWEN_IM_END}\n{QWEN_IM_START}user\n{input_}{QWEN_IM_END}\n{QWEN_IM_START}assistant\n"

        elif self.args.model_type == 'llama':
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{input_}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        
        else:
            raise ValueError(f"Model type {self.args.model_type} not supported!")

        return prompt
    
    def combine_prompt_2(self, sample):
        if self.args.task_type == 'pigai_yuwen':

            problem = sample['ocr'].replace('"','')
            prompt_template = """你是一位经验丰富的中国K12语文老师，你现在的任务是批改学生作答的语文题目。

【任务内容】
输入格式：<题目>为OCR识别到的试卷题目，题目中的学生答案以“##”符号分隔。需要提取出全部的学生答案，每个学生答案都要返回一个批改结果。你需要生成一个严格遵循格式的JSON格式<批改结果>，不进行任何自定义修改。

【输出规范】
1.输出的json格式包含“data”字段，该字段是一个包含所有题目的列表。列表中的每个json字典对应一个题目的批改结果。
2.每道题必须包含两个字段，分别是：“response”、“type”。
3.“response”字段中包含对这道题的全部小空的批改结果。每个小空都有以下三个字段：“hand_text”字段、“answer”字段和“result”字段。
    4.1.“hand_text”字段为学生答案。该字段的值是从“question”字段中“##”中提取出来的，例如，##A)## 提取到的内容为A，则“hand_text”字段的值为A。忽略学生答案中的特殊符号。
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
                    "answer": "B",
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
            input_ = prompt_template.format(question=problem)
        
        else:
            raise ValueError

        return input_

    def process(self):
        data_list = self.dataset
        # data_list = data_list[:5]

        if 'Qwen3' in self.args.sft_model_path:
            sub_inputs = list(map(self.combine_prompt_2, data_list))
            output_str_list = self.generate(sub_inputs)
        else:
            sub_inputs = list(map(self.combine_prompt, data_list))
            output_str_list = self.generate(sub_inputs)

        for idx in range(len(data_list)):
            data_list[idx].update(
                {
                    'model_output': output_str_list[idx]
                }
            )

        return data_list


if __name__ == "__main__":
    args = get_args()

    engine = LLMInference(args)

    print(
        f"*** Starting to generate {args.max_gen_length}, prompts={len(engine.dataset)}",
    )
    result_list = engine.process()
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    if args.save_path.endswith('jsonl'):
        save_jsonL(args.save_path, result_list)
    elif args.save_path.endswith('csv'):
        import pandas as pd
        df = pd.DataFrame(result_list)
        df.to_csv(args.save_path, index=False, encoding='utf-8-sig')
        # 直接使用 save_csv 函数
        #
        # save_csv(args.save_path, result_list)
    else:
        save_json(args.save_path, result_list)
