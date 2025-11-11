import json
import os
import requests
import numpy as np
import cv2
import base64
from openpyxl import load_workbook
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import time
import uuid

def numpy_to_base64(image_np):
    data = cv2.imencode('.jpg', image_np)[1]
    image_bytes = data.tobytes()
    image_base4 = base64.b64encode(image_bytes).decode('utf8')
    return image_base4


def search(img_b64):
    # trace_id = "04b1_35e5f6b1-1095-4b63-a22d-5cb167aa5d32"
    trace_id = uuid.uuid4().hex

    data = {}
    # img = cv2.imdecode(np.frombuffer(requests.get(img_url).content, np.uint8), cv2.IMREAD_COLOR)
    # data['img_base64'] = numpy_to_base64(img)
    

    data['img_base64'] = img_b64
    data["trace_id"] = trace_id

    # url = r'http://hmi.chengjiukehu.com/smart-match/search_page'
    url = 'http://t-talk.vdyoo.net/smart-search/search_page'
    

    headers = {
        'cache-control': "no-cache",
        'Postman-Token': "7116a046-e240-4710-a42c-8157fabbb71e"
    }

    res = requests.request("POST", url, data=json.dumps(data), headers=headers)

    res_json = json.loads(res.text)
    # print(res.text)
    # print(res_json['data'])
    try:
        res_ = res_json['data'][0]
    except IndexError:
        print(f"Error: No data found for trace_id {trace_id}")
        print(res_json)
        return {}
    return res_json['data'][0]

def main(args):
    # path = '/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data2025/0925jiaofu/zhengye'
    # jiaozheng_dir = '/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/jiaofu/db_data/0822-2/chailu/jiaozheng'
    img_path = args.img_path
    img_name = os.path.split(img_path)[-1]
    search_save_path = args.search_save_path
    compare_img_path = args.compare_img_path
    # save_dir = f'{path}/smart_match_results'
    # save_json_dir = f'{path}/smart_match_json'

    
    img = cv2.imread(img_path)
    img_b64 = numpy_to_base64(img)
    res = search(img_b64) ### 整页搜索结果
    
    rst_url = res['ans_img'] if res else ""
    print(img_path + '\n' +rst_url + '\n====================')
    with open(search_save_path, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)

    if rst_url:
        rst_img = cv2.imdecode(np.frombuffer(requests.get(rst_url).content, np.uint8), cv2.IMREAD_COLOR)
        rst_img = cv2.resize(rst_img, (img.shape[1], img.shape[0]))

        ### 拼接img和rst_img (原图 + 题库图)
        combined_img = np.hstack((img, rst_img))

        # 保存结果
        cv2.imwrite(compare_img_path, combined_img)

        # 打印结果
        print(f"Processed {img_name}, saved result to {compare_img_path}")

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--img_path', type=str, default='')
    argparser.add_argument('--search_save_path', type=str)
    argparser.add_argument('--compare_img_path', type=str)

    args = argparser.parse_args()

    main(args)

    # img_path='/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/jiaofu/db_data/3books-0808/54ye-smart_match/jiaozheng/7989_82a4b113-8ae9-4d12-9961-f68ad6abadec.jpg'

    # img = cv2.imread(img_path)
    # img_b64 = numpy_to_base64(img)
    # res = search(img_b64)
    # print(res)