# -*- coding: utf-8 -*-
"""
@date: 2020/4/30 2:49 下午
@desc：
    获取HTTP签名和发送HTTP请求Demo
    content-type: application/json
    python版本 > 3.0
 
"""
import uuid
import base64
import hmac
import time
import requests
import json, csv
import sys, os, re
import numpy as np
from hashlib import sha1
from urllib.parse import quote
from requests.exceptions import RequestException
import traceback
from base64 import b64encode
from PIL import Image, ImageDraw, ImageOps
# from get_ocr import get_origin_ocr
import cv2
from tqdm import tqdm

def remove_punctuation_and_special_chars(text):
    # 使用正则表达式匹配所有非汉字、非英文字母和非数字的字符并替换为空字符串
    special_rules = [
                        ("①", "1"),
                        ("②", "2"),
                        ("③", "3"),
                        ("④", "4"),
                        ("⑤", "5"),
                        ("⑥", "6"),
                        ("✓", "对"),
                        ("×", "错")
                    ]
    for i in special_rules:
        text = text.replace(i[0], i[1])
    return re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)

def base64_to_numpy(image_base64):
    image_bytes = base64.b64decode(image_base64)
    image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return image_np
def base64_to_image(base64_str):
    imgData = base64.b64decode(base64_str)
    nparr = np.fromstring(imgData, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def image2base64(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    img_base64 = b64encode(img_encoded)
    return img_base64.decode()

class ApplicationJsonRequest(object):
    def __init__(self, url, url_params, body_params, access_key_id, access_key_secret):
 
        # 设置请求头content-type
        self.headers = {'content-type': "application/json"}
 
        # 请求URL，请替换自己的真实地址
        self.url = url
 
        # 填写自己AK
        # 获取AK教程：https://openai.100tal.com/documents/article/page?fromWhichSys=admin&id=27
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
 
        # 根据接口要求，填写真实Body参数。key1、key2仅做举例
        self.body_params = body_params
 
        # 根据接口要求，填写真实URL参数。key1、key2仅做举例
        self.url_params = url_params
 
    @property
    def timestamp(self):
        # 获取当前时间（东8区）
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
 
    @staticmethod
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
 
    def _generate_signature(self, parameters, access_key_secret):
 
        # 计算证书签名
        string_to_sign = self.url_format(parameters)
 
        #  进行base64 encode
        secret = access_key_secret + "&"
        h = hmac.new(secret.encode('utf-8'), string_to_sign.encode('utf-8'), sha1)
        signature = base64.b64encode(h.digest()).strip()
        signature = str(signature, encoding="utf8")
        return signature
 
    def get_signature(self):
 
        self.url_params['access_key_id'] = self.access_key_id
        self.url_params['timestamp'] = self.timestamp
 
        # 组合URL和Body参数，并计算签名
        self.url_params['signature_nonce'] = str(uuid.uuid1())
 
        sign_param = {
            "request_body": json.dumps(self.body_params)
        }
        sign_param.update(self.url_params)
 
        signature = self._generate_signature(sign_param, self.access_key_secret)
 
        self.url_params['signature'] = quote(signature, 'utf-8')
 
 
    def run(self):
        # 生成签名
        self.get_signature()
 
        # 生成URL
        url = self.url + '?' + self.url_format(self.url_params)
        # 响应结果httpResponse
        try:
            response = requests.post(url, json=self.body_params, headers=self.headers, timeout=15)
        except RequestException:
            print(traceback.format_exc())
            return {'code':0, 'msg':traceback.format_exc()}
        # print("dfdfdf")
        # print(response)
        return response.json()
 
def get_topN_result(images):
    image_base4s = [image2base64(i) for i in images]
    # http_flag = 'http' in image_path
    # if not http_flag:
    #     # img = open(image_path, "rb").read()
    #     # image_path = b64encode(img).decode()
    #     im = cv2.imread(image_path, 0)
    #     im = cv2.imencode('.jpg', im)[1]
    #     image_path = b64encode(im).decode()


    # access_key_id = '1384834886006738944'
    # access_key_secret = '3d213a6b4a56487191eb1a47f63b096b'

    # qinglin key
    access_key_id = "1384834886006738944"
    access_key_secret = "3d213a6b4a56487191eb1a47f63b096b"

    # url = "https://openai.100tal.com/aiimage/comeducation"
    # url = "https://openai.100tal.com/aiimage/ocr-det-correction"
    
    # url = "http://gateway-bp-bd.facethink.com/aiimage/text-formula-union-pigai"
    url = "http://openai.100tal.com/aiimage/text-formula-union-pigai"
 
    url_params =  {'Content-Type': 'application/json'}
    # 根据接口要求，填写真实URL参数。key1、key2仅做举例
    
    body_params = {
        "image_base64s":image_base4s,
        "beam_width":5
    }
    res = ApplicationJsonRequest(url=url, access_key_id=access_key_id, access_key_secret=access_key_secret,
                        body_params=body_params, url_params=url_params).run()
    return res

def hand_text_info(ocr_res):
    '''
    ocr_res: 图片拆录结果
    '''
    outputs = []
    outputs.append(['hand_text', 'Confidence', 'std', 'vertices'])
    
    result = ocr_res['data']['result']
    for result_ in result:
        all_single_box = result_['all_single_box']
        for _all_single_box in all_single_box:
            if "hand_text" in _all_single_box:
                hand_text_list = _all_single_box['hand_text']
                for hand_box in hand_text_list:
                    outputs.append([hand_box['texts'], hand_box['Confidence'], hand_box['stdofconfi'], hand_box['poses']])
    print(len(outputs))
    return outputs

def get_ocr_with_topN(img_dir, ocr_res, save_dir):
    '''
    img_dir: 原始图片路径
    ocr_res: 原始图片的OCR拆录结果
    save_dir: 裁剪后每个box的保存路径
    '''
    os.makedirs(save_dir, exist_ok=True)

    
    # num = 0
    # result = ocr_res['data']['result']
    outputs = []
    for num, res in enumerate(tqdm(ocr_res)):
        if res['code'] != 20000:
            output = [num+1,res['img_url'], ""]
            continue
        result = res['data']['result']
        page_line = []
        current_title = ""
        img_url = res['img_url']
        img_name = os.path.split(img_url)[-1]
        for idx, result_ in enumerate(result):
            result_data = result_['data']
            for data_ in result_data:
                for i, item in enumerate(data_):
                    type = item['type']
                    # 拼接singleBox
                    text_new = item['text_new']
                    if text_new['singleBox'] is not None:
                        text_new_list = [box['text'] for box in text_new['singleBox']]
                        text = " ".join(text_new_list)
                        
                        if type == 'title':
                            current_title = text
                            continue
                        if len(current_title) > 0 :
                            text = current_title + " " + text ### 将标题与题干信息拼接
                        vertices = item['expand_quad_location'] ### 边框顶点坐标

                        img_path = f"{img_dir}/{img_name}"
                        # 打开原始图像
                        image = Image.open(img_path)
                        ### top-N
                        topN_all = {}
                        for box in text_new['singleBox']:
                            if not box['is_print']:
                                x = box['x']
                                y = box['y']
                                cropped_image = image.crop((x, y, x+box['width'], y+box['height']))
                                cropped_image_np = np.array(cropped_image)
                                ocr_topN = get_topN_result([cropped_image_np])
                                clean_text = remove_punctuation_and_special_chars(box['text'])
                                topN_all.update({
                                    clean_text: ocr_topN['data']['result'][0]
                                })
                                # topN_all.append(topN_list)
                                # cropped_image.save(f"{save_dir}/box_{box['text']}.jpg")
                        if idx == 0:
                            page_line.append([num+1, res['img_url'], str(vertices), current_title, text, topN_all])
                        else:
                            page_line.append(["", "", str(vertices), current_title, text, topN_all])
        outputs += page_line

    return outputs



if __name__ == "__main__":

    # path = sys.argv[1]

    # img_path='/mnt/pfs/jinfeng_team/APP/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/yuwen/1015-topN/ca254104-1fd0-4586-80b2-315a2ab04ea4.jpg'
    # ocr_path='/mnt/pfs/jinfeng_team/APP/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/yuwen/1015-topN/ca254104-1fd0-4586-80b2-315a2ab04ea4.jpg.json'
    # ocr_res = json.load(open(ocr_path,'r'))
    # save_dir = '/mnt/pfs/jinfeng_team/APP/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/yuwen/1015-topN/crop'
    # path = '/mnt/pfs/jinfeng_team/APP/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/yuwen/1025-400'
    # img_dir = f'{path}/jiaozheng'
    # ocr_path = f"{path}/ocr_supp.json"
    # save_dir2 = f'{path}/topN_crop'

    # ocr_res = json.load(open(ocr_path,'r'))
    # outputs = get_ocr_with_topN(img_dir, ocr_res, save_dir2)

    # save_csv = f'{path}/ocr_origin_chaiti_topN.csv'
    # with open(save_csv, 'w', encoding='utf-8', newline="") as f:
    #     w=csv.writer(f)
    #     w.writerow(['序号', 'url', 'vertices', 'title', 'ocr', 'topN'])
    #     w.writerows(outputs)

    img_path = 'Correct_model/pigai_pipeline/pigai_v1/data/0108-4badcase/0306local/jiaozheng/98213327-6409-4c2e-a2e5-f10a0a584606.jpg'
    image = Image.open(img_path)
    cropped_image_np = np.array(image)
    ocr_topN = get_topN_result([cropped_image_np])
    print(ocr_topN)
    