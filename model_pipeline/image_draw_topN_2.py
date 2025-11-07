import os, csv ,json, sys
import requests
from PIL import Image, ImageDraw, ImageOps
from pathlib import Path
from .json_match_inaccurate_topN_2 import match_inaccurate, remove_latex_commands

from ..utils.utils import load_json, load_csv, load_jsonL
import ast

def draw_check_mark(draw_obj, x, y):
    length = 10
    draw_obj.line((x,y+20, x-length,y), fill='green', width=6)
    draw_obj.line((x,y+20, x+length+10,y-5), fill='green', width=6)

def draw_cross(draw_obj,x,y):
    x=x+10
    length = 10
    draw_obj.line((x-length,y-length,x+length,y+length),fill="red", width=6)
    draw_obj.line((x+length,y-length,x-length,y+length),fill="red", width=6)

def draw(img_path, data, save_img_path, rotate_angle=0):

    # 打开本地图片
    img = Image.open(img_path)

    # 消除自动旋转
    img = ImageOps.exif_transpose(img)
    img = img.rotate(rotate_angle, expand=True)

    # 创建绘图对象
    draw = ImageDraw.Draw(img)

    # 遍历坐标信息并绘制对勾或叉
    for item in data:
        result = item['result']
        for box in item['matched_boxes']:
            x = box['x']
            y = box['y']
            if result == 1:
                draw_check_mark(draw,x + box['width']//2,y + box['height']//2)
            elif result == 0:
                draw_cross(draw,x + box['width']//2,y + box['height']//2)
    # print(save_img_path)

    img.save(save_img_path)

def main(args):
    model_output_path = args.model_output ### 模型批改结果
    jiaozheng_path = args.jiaozheng_path ### 原图路径

    pigai_save_path = args.pigai_save_path
    draw_path = args.draw_path

    json_output = load_csv(model_output_path)
    pt = 1
    colnames = json_output[0]
    print(colnames)
    hand_text_list_index = colnames.index('hand_text_list')
    coordinate_index = colnames.index('vertices')
    url_index = colnames.index('img_url')
    output = []
    draw_output = []
    for i, line in enumerate(json_output[1:]):
        img_url = line[url_index]
        image_name = os.path.split(img_url)[-1]
        image_path=jiaozheng_path

        correct_response = line[-1] ### 模型输出json
        coordinate = line[coordinate_index] ### 题框坐标
        
        hand_text_list = ast.literal_eval(line[hand_text_list_index]) ### 手写体列表

        tmp_draw = [] ### 当前行坐标
        if correct_response:
            
            result = match_inaccurate(correct_response, hand_text_list, coordinate) ## 模糊匹配批改结果
            output += result
            for j in result:
                for k in j['matched_boxes']:
                    x = k['x']+k['width']//2
                    y = k['y']+k['height']//2
                    x1 = k['x']+k['width']
                    y1 = k['y']+k['height']
                    tmp_draw.append({"x":x, "y":y, "result":j['result'], "box":[k['x'], k['y'], x1,y1]})
        draw_output.append([img_url, tmp_draw])
    
    save_img_path = f'{pigai_save_path}/{image_name}'
    if not (save_img_path.endswith('jpg') or save_img_path.endswith('png')):
        save_img_path = f"{save_img_path}.jpg"

    if not Path(save_img_path).parent.exists():
        Path(save_img_path).parent.mkdir(parents=True, exist_ok=True)
    draw(image_path, output, save_img_path, rotate_angle=0)
        
    print(f"pigai image saved to {save_img_path}")

    merge_data = {}
    for i in draw_output:
        if i[0] in merge_data:
            merge_data[i[0]] += i[1]
        else:
            merge_data[i[0]] = i[1]
    merge_data_1 = [[url, match_boxes] for url, match_boxes in merge_data.items()]
    with open(draw_path, 'w', encoding='utf-8', newline="") as f:
        w = csv.writer(f)
        w.writerows(merge_data_1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Draw pigai results")
    parser.add_argument("--model_output", type=str, required=True, help="model_output.csv")
    parser.add_argument("--jiaozheng_path", type=str, required=True, help="Local image path.")
    parser.add_argument("--pigai_save_path", type=str, required=True, help="Pigai save path.")
    parser.add_argument("--draw_path", type=str, required=True, help="draw.csv")
    args = parser.parse_args()

    main(args)



