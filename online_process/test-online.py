import cv2
import numpy as np
import csv
from PIL import Image, ImageDraw, ImageOps, ImageFont
import json
import random
import re
import json, csv, os, sys
import itertools, re, difflib
from Levenshtein import distance
import base64, urllib.request

from tqdm import tqdm

font = ImageFont.truetype("/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/utils/SimHei.ttf", 15)

def download_image(url, save_path):
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"图片成功保存在 {save_path}")
    except Exception as e:
        print(f"图片下载失败: {e}")

def load_csv_2_dict(csv_path):
    data = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        col_names = next(reader)
        csv_reader = csv.DictReader(f, col_names)
        for row in csv_reader:
            d = {}
            for k,v in row.items():
                d[k]=v
            data.append(d)
            # if len(data) == 1002:
            #     break

    return data

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_jsonL(json_path):
    with open(json_path, 'r') as f:
        dataset = f.readlines()
    # data = [json.loads(d.strip()) for d in dataset]
    data = []
    for d in dataset:
        try:
            data.append(json.loads(d.strip()))
        except:
            print(d)
    return data

def draw_check_mark(draw_obj, x, y):
    length = 10
    draw_obj.line((x,y+20, x-length,y), fill='green', width=6)
    draw_obj.line((x,y+20, x+length+10,y-5), fill='green', width=6)

def draw_cross(draw_obj,x,y):
    x=x+10
    length = 10
    draw_obj.line((x-length,y-length,x+length,y+length),fill="red", width=6)
    draw_obj.line((x+length,y-length,x-length,y+length),fill="red", width=6)

def match_inaccurate_online_result(online_res, output_path):
    outputs = []
    for json_ in tqdm(online_res):
        if 'test_result' in json_:
            data = json_["test_result"]["data"]
        else:
            data = json_["data"]
        # data = json_["test_result"]["data"]
        if 'url' not in json_:
            continue
        img_name = json_["url"].split("/")[-1]
        # subject = json_["test_result"]["subject"]
        # if subject not in ['语文','语']:
        #     print(f"{img_name}: wrong classification! {subject}")
        #     continue
        if not data:
            continue
        else:
            data1 = data["results"] if "results" in data else None
            if not data1:
                continue

        matched_results = []
        for question in data1:
            # print(question["judgment_result"])
            results = question["judgment_result"]
            if not results:
                continue
            results_position = question["judgment_result_position"]
            if results_position == None or len(results_position) == 0:
                continue
            for i in range(len(results)):
                box = {}
                temp_single_boxes = []
                result = results[i]
                try:
                    box["x"] = results_position[i][0]
                    box["y"] = results_position[i][1]
                except:
                    print(img_name)
                    print(results_position)
                # box["width"]=results_position[i][2]
                # box["heigt"]=results_position[i][3]
                box["width"] = results_position[i][2] - results_position[i][0]
                box["height"] = results_position[i][3] - results_position[i][1]
                x = box['x']+box['width']//2
                y = box['y']+box['height']//2
                matched_results.append({"x":x, "y":y, "result": 1 if result=='正确' else 0})

        outputs.append([img_name, matched_results])
    print("outputs len: ", len(outputs))
    with open(output_path, 'w', encoding='utf-8', newline="") as f:
        w=csv.writer(f)
        w.writerows(outputs)


def draw(img_path, data, save_img_path, rotate_angle=0):
    # 打开本地图片
    # img = Image.open(img_path)
    try:
        # 检查文件是否存在
        if os.path.exists(img_path):
            # 尝试打开图像文件
            img = Image.open(img_path)
            # print("图像已成功打开并显示")
        else:
            raise FileNotFoundError(f"The file at path {img_path} does not exist.")

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        return 

    # 消除自动旋转
    img = ImageOps.exif_transpose(img)
    img = img.rotate(rotate_angle, expand=True)

    # 创建绘图对象
    draw = ImageDraw.Draw(img)

    # 遍历坐标信息并绘制对勾或叉
    for item in data:
        result = item['result']
        source = item['source']
        # print(result)
        for box in item['matched_boxes']:
            # print(box)
            x = box['x']
            y = box['y']
            draw.text((x,y), source, fill='blue', font=font)
            if result == "正确":
                draw_check_mark(draw,x + box['width']//2,y + box['height']//2)
            elif result == "错误":
                draw_cross(draw,x + box['width']//2,y + box['height']//2)

    # 保存或显示处理后的图片
    # img.show() # 或者 img.save("output.jpg")
    if not (save_img_path.endswith('jpg') or save_img_path.endswith('png')):
        img.save(f"{save_img_path}.jpg")
    else:
        img.save(save_img_path)



if __name__ == "__main__":
    # path = '/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/0904data'
    # path = '/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data2025/1027chailu-test/online'
    
    # k = sys.argv[1]

    # online_res = f"{path}/1027晚上-指标验证_yuwen_url_yuwen_yewu_0.jsonl"
    # online_res = f"/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data2025/1027chailu-test/语文批改结果/批改_语文_线上1027/1027晚上-指标验证_yuwen_url_yuwen_yewu_{k}.jsonl"
    # path = f"/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data2025/1027chailu-test/yewu_data-online/idx_{k}"
    
    online_res = '/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/0904yewu/1103test/online/1104下午1643_线上_流式_chinese_isangleFalse.jsonl'
    path = os.path.dirname(online_res)

    os.makedirs(path, exist_ok=True)
    online_res = load_jsonL(online_res)
    output_path = f"{path}/draw.csv"
    match_inaccurate_online_result(online_res, output_path)

    # input_file = "/mnt/pfs/jinfeng_team/APP/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/yuwen/xuexiji-40-1106_online/test学习机_result.jsonl"
    # image_dir = f"{path}/jiaozheng"
    image_dir = '/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data2025/1027chailu-test/online/jiaozheng'
    os.makedirs(image_dir, exist_ok=True)
    out_dir = f"{path}/pigai"
    os.makedirs(out_dir, exist_ok=True)
    # jsons = load_jsonL(input_file)

    for json_ in tqdm(online_res):
        if 'test_result' in json_:
            data = json_["test_result"]["data"]
            
        else:
            data = json_["data"]
        if 'url' not in json_:
            continue
        img_url = json_["url"]
        img_name = json_["url"].split("/")[-1]
        print(img_name)
        image_path = os.path.join(image_dir, img_name)
        if not os.path.exists(image_path):
            download_image(img_url, image_path)
        save_img_path = os.path.join(out_dir, img_name)
        if data == None:
            continue
        else:
            data1 = data["results"] if "results" in data else []
            if not data1:
                continue
        # print(img_name)
        matched_results = []
        for question in data1:
            # print(question["judgment_result"])
            results = question["judgment_result"]
            if results == None:
                continue
            results_position = question["judgment_result_position"]
            results_source = question["judgment_source"]
            if results_position == None or len(results_position) == 0:
                continue
            
            for i in range(len(results)):
                box = {}
                temp_single_boxes = []
                result = results[i]
                source = results_source[i]
                box["x"] = results_position[i][0]
                box["y"] = results_position[i][1]
                # box["width"]=results_position[i][2]
                # box["heigt"]=results_position[i][3]
                box["width"] = results_position[i][2] - results_position[i][0]
                box["height"] = results_position[i][3] - results_position[i][1]

                temp_single_boxes.append(box)

                matched_results.append({
                    'result': result,
                    'matched_boxes': temp_single_boxes,
                    'source': source
                })
                # print(temp_single_boxes)
        # print(matched_results)
        draw(image_path, matched_results, save_img_path, rotate_angle=0)
        # exit()