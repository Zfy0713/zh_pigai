#coding:utf-8
# from tkinter.tix import Tree
import time
from io import BytesIO
import base64
import uuid
import hmac
from hashlib import sha1
import uuid
from urllib import parse
from urllib.parse import quote
import requests
from urllib.parse import quote
import os, csv, json, re
from tqdm import tqdm

# 测试环境
# ACCESS_KEY_ID = "4786117720392704" # AI_sulotion_pre_test
# ACCESS_KEY_SECRET = "0c5501d4f1a84d508b590846542cdceb"

### 手机账户
# ACCESS_KEY_ID = "1338526759263404032"
# ACCESS_KEY_SECRET = "1df8b10e2c394a408f72208ce926e3b7"

### 知音楼账户
ACCESS_KEY_ID = "1384834886006738944"
ACCESS_KEY_SECRET = "3d213a6b4a56487191eb1a47f63b096b"
# HTTP_ANSWER_EXTRACT_URL = "http://gateway-bp.facethink.com/aitext/answer-extract/answer-extract/http"
HTTP_ANSWER_EXTRACT_URL = "http://openai.100tal.com/aitext/answer-extract/answer-extract/http"


def url_format(params):
        """
        # 对params进行format
        # 对 params key 进行从小到大排序
        :param params: dict()
        :return:
        a=b&c=d
        """
        sorted_parameters = sorted(params.items(), key=lambda d: d[0], reverse=False)
 
        param_list = ["{}={}".format(key, value) for key, value in sorted_parameters]
 
        string_to_sign = '&'.join(param_list)
        return string_to_sign

def get_signature(url_params, body_params, access_key_secret):
    # 生成请求    
    def _generate_signature(parameters, access_key_secret):
        string_to_sign = url_format(parameters)
        secret = access_key_secret + "&"
        h = hmac.new(secret.encode('utf-8'), string_to_sign.encode('utf-8'), sha1)
        signature = base64.b64encode(h.digest()).strip()
        signature = str(signature, encoding="utf8")
        return signature
    signature_nonce = str(uuid.uuid1())
    __request_body = "request_body"
    sign_param = {'signature_nonce': signature_nonce, __request_body: json.dumps(body_params)}
    for key in url_params.keys():
        sign_param[key] = url_params[key]
    signature = _generate_signature(sign_param, access_key_secret)
    return signature, signature_nonce

def my_request(url, url_params, body_params):
    header = 'application/json'
    access_key_id = ACCESS_KEY_ID
    access_key_secret = ACCESS_KEY_SECRET
    # 获取当前时间（东8区）
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    url_params['access_key_id'] = access_key_id
    url_params['timestamp'] = timestamp

    signature, signature_nonce = get_signature(
        url_params,
        body_params,
        access_key_secret)

    url_params['signature'] = quote(signature, 'utf-8')
    url_params['signature_nonce'] = signature_nonce
    param_list = []
    for key, value in url_params.items():
        param_str = '{}={}'.format(key, value)
        param_list.append(param_str)
    string_to_sign = '&'.join(param_list)
    url = url + '?' + string_to_sign
    headers = {
        'content-type': header
    }
    response = requests.post(url, json=body_params, headers=headers)
    # print(response.content.decode("utf-8"))
    # print(response.text)
    # return json.loads(response.content.decode("utf-8"))
    return json.loads(response.content)

# 答案抽取
def req_answer_extract(question = '', answer = ''):
    url_params =  {'Content-Type': 'application/json'}
    if len(question) == 0:
        print('question: ', question)
        return 
        raise ValueError("question is None!")
    else:
        body_params = {
            "messages":[
                {
                    "role": "user",
                    "question": question,
                    "answer":answer
                }
            ]
        }
    # try:
    while True:
        res = my_request(HTTP_ANSWER_EXTRACT_URL,url_params,body_params)
        # print(res)
        if res['code'] == 20000:
            break
        else:
            time.sleep(0.5)
    res = res['data']['result']
    return res

def answer_extract(item):
    '''
    答案抽取模型
    item:  题干+手写体+题库搜索结果
    return: 答案列表（3组答案）
    '''
    ocr = item['text_vl'].strip()
    ocr = re.sub(r'##.*?##', '', ocr)
    title = item['title'].strip()
    if ocr.startswith(title):
        ocr = ocr[len(title):].strip()
    
    rag_result = item['combined_rag']
    if rag_result == 'xx':
        return 
    if isinstance(rag_result, str):
        rag_result = ast.literal_eval(rag_result)

    def get_rag_res(rank_rag):
        if not rank_rag:
            return
        sim = float(rank_rag['similarity'])
        # if len(clean_text(ocr)) > 300: ## 长文本/阅读题不用题库批改
        #     return
        if rank_rag['source'] == 'tusou':
            THRESHOLD=0.95
        else:
            THRESHOLD=0.9
        
        if sim < THRESHOLD: ### 相同题判断，低于阈值的认为不是相同题。
            return
        else:
            rag_res = req_answer_extract(question = rank_rag['question'], answer = rank_rag['answer']) if rank_rag['source'] == 'tusou' else \
                req_answer_extract(question = rank_rag['question'], answer = rank_rag['analysis']+'\n'+rank_rag['answer'])
            time.sleep(0.5)
            return rag_res

    rag_res_all = list(map(get_rag_res, rag_result))
    cleaned_res = [str(item) for item in rag_res_all if item]
    return cleaned_res


if __name__ == '__main__':
    # 答案抽取
    item = {"img_url": "https://ss-prod-genie.oss-cn-beijing.aliyuncs.com/correct_pipeline/processed_image/2025-09-16/fa9c5ca7-573e-4946-993d-7b3cf94f1e28.jpg", "vertices": [{"x": 99, "y": 732}, {"x": 1636, "y": 736}, {"x": 1635, "y": 1127}, {"x": 98, "y": 1123}], "title": "二、根据语境,看拼音,写字词。(目标:201)", "text": "shuāng tíng hú xié xié 霜 降那天下午,我和妈妈来到洞 ##庭湖## 边,沿着 ##斜斜## 的 xiǎo jìng luò ##小径## 散步。我捡了一片自己喜爱的 ##落## 叶带回家。", "VL_response": "$\\overset{shuāng}{\\tianzige{<ans><ref>霜</ref><box>[[61, 190, 116, 477]]</box></ans>}}$降那天下午，我和妈妈来到洞$\\overset{tíng}{\\tianzige{<ans><ref>庭</ref><box>[[523, 203, 577, 423]]</box></ans>}}$\\overset{hú}{\\tianzige{<ans><ref>湖</ref><box>[[601, 225, 663, 432]]</box></ans>}}$边，沿着$\\overset{xié}{\\tianzige{<ans><ref>斜</ref><box>[[805, 220, 866, 445]]</box></ans>}}$\\overset{xié}{\\tianzige{<ans><ref>斜</ref><box>[[879, 220, 945, 455]]</box></ans>}}$的\n$\\overset{xiǎo}{\\tianzige{<ans><ref>小</ref><box>[[20, 725, 65, 914]]</box></ans>}}$\\overset{jìng}{\\tianzige{<ans><ref>径</ref><box>[[84, 725, 143, 957]]</box></ans>}}$散步。我捡了一片自己喜爱的$\\overset{luò}{\\tianzige{<ans><ref>落</ref><box>[[504, 676, 562, 960]]</box></ans>}}$叶带回家。", "VL_decode_res": {"answer_list": [{"answer": "霜", "bbox": [[93, 75], [178, 188]], "label": 0}, {"answer": "庭", "bbox": [[804, 80], [887, 167]], "label": 1}, {"answer": "湖", "bbox": [[924, 88], [1019, 170]], "label": 2}, {"answer": "斜", "bbox": [[1238, 86], [1331, 175]], "label": 3}, {"answer": "斜", "bbox": [[1351, 86], [1453, 179]], "label": 4}, {"answer": "小", "bbox": [[30, 286], [99, 361]], "label": 5}, {"answer": "径", "bbox": [[129, 286], [219, 378]], "label": 6}, {"answer": "落", "bbox": [[775, 267], [864, 379]], "label": 7}], "image_list": [], "question": "$\\overset{shuāng}{\\tianzige{}}$降那天下午，我和妈妈来到洞$\\overset{tíng}{\\tianzige{}}$\\overset{hú}{\\tianzige{}}$边，沿着$\\overset{xié}{\\tianzige{}}$\\overset{xié}{\\tianzige{}}$的\n$\\overset{xiǎo}{\\tianzige{}}$\\overset{jìng}{\\tianzige{}}$散步。我捡了一片自己喜爱的$\\overset{luò}{\\tianzige{}}$叶带回家。", "simple_response": "$\\overset{shuāng}{\\tianzige{霜}}$降那天下午，我和妈妈来到洞$\\overset{tíng}{\\tianzige{庭}}$\\overset{hú}{\\tianzige{湖}}$边，沿着$\\overset{xié}{\\tianzige{斜}}$\\overset{xié}{\\tianzige{斜}}$的\n$\\overset{xiǎo}{\\tianzige{小}}$\\overset{jìng}{\\tianzige{径}}$散步。我捡了一片自己喜爱的$\\overset{luò}{\\tianzige{落}}$叶带回家。"}, "hand_text_list": [{"height": 113, "is_print": False, "text": "##霜##", "width": 85, "x": 191, "y": 807, "topN": {"results": [{"text": " 霜", "prob_reg": 0.999508}, {"text": " 霖", "prob_reg": 0.00022346628}, {"text": " 箱", "prob_reg": 0.00018817748}, {"text": " 霏", "prob_reg": 1.9989318e-05}, {"text": " 霸", "prob_reg": 1.9679414e-05}]}}, {"height": 84, "is_print": False, "text": "##庭湖##", "width": 212, "x": 903, "y": 815, "topN": {"results": [{"text": " 庭 湖", "prob_reg": 0.99998015}, {"text": " 底 湖", "prob_reg": 0.0046672393}, {"text": " 庄 湖", "prob_reg": 0.0025574362}, {"text": " 座 湖", "prob_reg": 0.0015941634}, {"text": " 庆 湖", "prob_reg": 0.0012909928}]}}, {"height": 86, "is_print": False, "text": "##斜斜##", "width": 212, "x": 1337, "y": 822, "topN": {"results": [{"text": " 斜 斜", "prob_reg": 0.99995524}, {"text": " 余 斗 斜", "prob_reg": 0.036998607}, {"text": " 斜 科", "prob_reg": 0.0042164996}, {"text": " 舒 斜", "prob_reg": 0.002402489}, {"text": " 科 斜", "prob_reg": 0.0016256054}]}}, {"height": 90, "is_print": False, "text": "##小径##", "width": 193, "x": 127, "y": 1016, "topN": {"results": [{"text": " 小 径", "prob_reg": 0.99999434}, {"text": " 小 经", "prob_reg": 0.0022569296}, {"text": " 小 征", "prob_reg": 0.0016512049}, {"text": " 小 行", "prob_reg": 0.0012222874}, {"text": " 小 轻", "prob_reg": 0.00097449246}]}}, {"height": 110, "is_print": False, "text": "##落##", "width": 92, "x": 872, "y": 999, "topN": {"results": [{"text": " 落", "prob_reg": 0.9995462}, {"text": " 蓉", "prob_reg": 0.0004240645}, {"text": " 蓼", "prob_reg": 1.14789e-05}, {"text": " 茗", "prob_reg": 8.664728e-06}, {"text": " 溶", "prob_reg": 3.1381567e-06}]}}], "text_vl": "shuāng tíng hú xié xié 霜 ##霜## 降那天下午,我和妈妈来到洞 ##庭湖## 边,沿着 ##斜斜## 的 xiǎo jìng luò ##小径## 散步。我捡了一片自己喜爱的 ##落## 叶带回家。", "diff_vl": True, "combined_rag": [{"rank": "0", "similarity": "0.9999256134033203", "question": "二、根据语境，看拼音，写字词。(目标:201)shuāng tíng hú xié xié降那天下午，我和妈妈来到洞 边,沿着 的xiǎo jìng luò散步。我捡了一片自己喜爱的 叶带回家。", "answer": "二、霜 庭湖 斜斜 小径 落", "analysis": "", "source": "tusou"}, {"rank": "1", "similarity": "0.9999256134033203", "question": "二、根据语境,看拼音,写字词。(目标:201)shuāng tíng hú xié xié降那天下午，我和妈妈来到洞 边,沿着 的xiǎo jìng luò散步。我捡了一片自己喜爱的 叶带回家。", "answer": "二、霜 庭湖 斜斜 小径 落", "analysis": "", "source": "tusou"}, {"rank": "2", "similarity": "0.9999209642410278", "question": "二、根据语境，看拼音，写字词。(目标：201) shuang  ting hu  xie xie 降那天下午，我和妈妈来到洞 边，沿着 的 xiao jing  luo 散步。我捡了一片自己喜爱的 叶带回家。", "answer": "二、霜庭湖 斜斜小径落", "analysis": "", "source": "tusou"}]}
    res = answer_extract(item)
    print(res)
    # print(res['data']['result'])
    