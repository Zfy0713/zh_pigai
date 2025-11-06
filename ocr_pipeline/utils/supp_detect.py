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

def supp_detect(image_path):
    '''
    image_path: 图片路径
    return: 作答区域补充结果
    '''
    access_key_id = "1384834886006738944"
    access_key_secret = "3d213a6b4a56487191eb1a47f63b096b"

    base64_data = base64.b64encode(open(image_path, "rb").read()).decode()
    body_params = {"image_base64": base64_data}
    url_params =  {'Content-Type': 'application/json'}
    url = "http://openai.100tal.com/aiimage/answer-supplementary-detect"
    request = ApplicationJsonRequest(url, url_params, body_params, access_key_id, access_key_secret)
    res = request.run()
    return res


if __name__ == "__main__":
    image_path = sys.argv[1]
    res = supp_detect(image_path)
    print(res)

    # /mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data2025/0418bei/chailu/jiaozheng/0a66_3f463623-af70-4b68-ba49-047aacca1cc1.jpg