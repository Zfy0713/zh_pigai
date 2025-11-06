import json, csv

import re
import os
import uuid
from tqdm import tqdm
import io
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from collections import OrderedDict
import base64
import requests
import pandas as pd
from utils.ocr_topN import get_topN_result
from utils.merge_vl_overset import merge_overset


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
    return data

class VL_pigai_OCR:
    """
    这是一个OCR识别的类，包含调用模型、解析输出等功能。
    """
    
    def __init__(self):
        self.FontPath='/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/utils/SimHei.ttf'
        self.Font = ImageFont.truetype(self.FontPath, 10)
        
    def smart_font_size(self, text, max_width, font_path, min_fontsize=10):
        if font_path is None:
            font_path = self.FontPath

        img = Image.new('RGB', (max_width, 100))  # 创建临时画布
        draw = ImageDraw.Draw(img)
        max_fontsize = 100
        fontsize = max_fontsize
        min_fontsize=min_fontsize
        while fontsize >= min_fontsize:
            font = ImageFont.truetype(font_path, fontsize)
            bbox = draw.textbbox((0, 0), text, font=font)
            # text_width, _ = font.getsize(text)
            text_width = bbox[2] - bbox[0]
            if text_width <= max_width:
                return font
            fontsize -= 1
        # 如果没有找到合适的字体大小，则返回最小字体大小
        return ImageFont.truetype(font_path, min_fontsize)
        

    def img_to_base64(self, image_path):
        """
        将图片转换为base64编码。
        :param image_path: 图片的路径
        :return: base64编码的字符串
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def vl_1b(self, img_b64, env='test'):
        if env == 'test':

            api_url = "https://mlops-infer.tal.com/appset/slave-xxj-paizhao-1b-wangm/v1/tal/completions" ## test env
            headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer a586835303105:zxcdb7b94c3b'} ### test env
        elif env == 'online':
            api_url = "https://mlops-infer.tal.com/appset/orc-xxj-paizhao-1b-wangm/v1/tal/completions" ## online env
            headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer a253788754481:pm2k2AHCld32'} ### online env



        request_id = uuid.uuid4().hex
        stream = False
        
        images = [img_b64]
        prompt = {"mm_data":{"type":"base64", "image": images}}
        
        body = {"request_id": request_id, "prompt": prompt, "stream": stream}
        try:
            response = requests.post(api_url, headers=headers, json=body, stream=stream)
        # print(response.content.decode('utf-8'))
        except Exception as e:
            print(f"Error during request: {e}")
            return f"Error during request: {e}"
        #解析返回的json数据
        outputs = []
        result = ""
        if stream:
            # 解析流式请求，每次返回一个token
            for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\n"):
                if chunk:
                    try:
                        data = json.loads(chunk.decode("utf-8").replace("data:", ""))
                        outputs.append(data["choices"][0]["text"])
                    except Exception as e:
                        pass
            result = "".join(outputs)
        else:
            # 解析非流式请求，一次性返回所有token
            try:
                data = json.loads(response.content)
                result = data["choices"][0]["text"]
            except Exception as e:
                print(f"Error parsing response: {response.content}\n{e}")
                return None
        # print(data)
        return result

    def bbox_norm2regular(self, bbox: list, image_path=None, width=None, height=None):
        try:
            bbox = [int(i) for i in bbox.split(',')]
        except Exception as e:
            print(f"bbox: {bbox}")
            print(f"Error converting bbox to int: {e}")
            return []
        assert len(bbox) >= 4 and len(bbox)%2==0
        try:
            if image_path is None:
                if width is None or height is None:
                    raise ValueError("Either image_path or both width and height must be provided.")
            else:
                # 如果提供了图片路径，则从图片中获取宽度和高度
                if width is None or height is None:
                    original_image = Image.open(image_path)
                    width, height = original_image.size
            # original_image = Image.open(image_path)
            # width, height = original_image.size
            bbox_regular = []
            for i in range(0, len(bbox), 2):
                x = int(bbox[i] / 1000 * width)
                y = int(bbox[i+1] / 1000 * height)
                bbox_regular.append([x,y])
            return bbox_regular
        except Exception as e:
            print(f"something wrong just happened : {e}")
    

    def extract_all_refs_and_boxes(self, response: str) -> list:
        """
        从响应中提取所有的引用和框。
        :param response: 响应字符串
        :return: 包含引用和框的列表
        """
        ans_blocks = re.findall(r"<ans>(.*?)</ans>", response, re.DOTALL)
        all_results = []
        # 对每个<ans>块，提取其中所有的<ref>和<box>对
        for ans_block in ans_blocks:
            ref_box_pairs = re.findall(r"<ref>(.*?)</ref><box>\[{1,2}(.*?)\]{1,2}</box>", ans_block, re.DOTALL)
            # if len(ref_box_pairs) > 1:
            #     ### 同一个<ans>块中有多个<ref>和<box>对，进行合并
            #     combined_ref = ''.join([pair[0] for pair in ref_box_pairs])
            #     # ref_box_pairs_combinetext = [(combined_ref ,pair[1]) for pair in ref_box_pairs]

            #     all_results.append(ref_box_pairs_combinetext)
            # else:
            all_results.extend(ref_box_pairs)
        return all_results

    def extract_puretext(self, response: str) -> str:
        ans_pattern_complete = r"<ans>.*?</ans>"
        puretext = re.sub(ans_pattern_complete, "", response, flags=re.DOTALL)
        return puretext.strip()
    
    def extract_simpletext(self, response: str) -> str:
        def extract_refs(match):
            ans_content = match.group(1)
            refs = re.findall(r"<ref>(.*?)</ref>", ans_content, re.DOTALL)
            return "".join(refs)
        
        simpletext = re.sub(r"<ans>(.*?)</ans>", extract_refs, response, flags=re.DOTALL)
        return simpletext.strip()

    def extract_internvl_response(self, response: str, image_path, width, height) -> dict :
        ans_pattern = r"<ans><ref>(.*?)</ref><box>\[{1,2}(.*?)\]{1,2}</box></ans>"
        img_pattern = r"<ref><-img.(\d+)->(.*?)</ref><box>\[{1,2}(.*?)\]{1,2}</box>"
        # ans_list = re.findall(ans_pattern, response, re.DOTALL)
        ans_list = self.extract_all_refs_and_boxes(response) ### 适配<ans></ans>中有多个作答区域
        img_list = re.findall(img_pattern, response, re.DOTALL)

        # puretext = re.sub(ans_pattern, "", response, flags=re.DOTALL)
        puretext = self.extract_puretext(response)
        puretext = re.sub(img_pattern, r"<-img.\1>", puretext, flags=re.DOTALL)

        # simpletext = re.sub(ans_pattern, r"\1", response, flags=re.DOTALL)
        simpletext = self.extract_simpletext(response)
        simpletext = re.sub(img_pattern, r"<-img.\1>\2", simpletext, flags=re.DOTALL)
        save_info = {
            "answer_list":[{"answer": ans_list[i][0], "bbox": self.bbox_norm2regular(ans_list[i][1], image_path, width, height), "label": i} for i in range(len(ans_list))],
            "image_list":[{"answer": img_list[i][1], "bbox": self.bbox_norm2regular(img_list[i][2], image_path, width, height), "label": int(img_list[i][0])} for i in range(len(img_list))],
            "question":puretext,
            "simple_response":simpletext
        }
        return save_info


class Merge_VL_Chailu(VL_pigai_OCR):
    '''
    合并试卷拆录和vl模型的结果
    '''
    def __init__(self, img_path, ocr_res, merge_overset_flag=False):

        super().__init__()
        self.img_path = img_path
        self.ocr_res = ocr_res
        self.image = Image.open(self.img_path)
        self.merge_overset_flag = merge_overset_flag

    def __call__(self):
        '''
        :param ocr_res: dict, 单张图试卷拆录结果
        :return: dict, 合并后的结果
        '''

        image = self.image
        ocr_res = self.ocr_res
        img_url = ocr_res['img_url']
        result = ocr_res['data']['result']
        output = []
        all_hand_list = []
        current_title = ""
        
        for idx, result_ in enumerate(result):
            result_data = result_['data']
            for data_i, data_ in enumerate(result_data):
                for item_i, item in enumerate(data_):
                    type = item['type']
                    text_new = item['text_new']
                    
                    if text_new['singleBox'] is not None:

                        chailu_singlebox_list = self.add_topN_into_singlebox_list(text_new['singleBox']) ### 对拆录的手写体添加topN结果
                        text_new_list = [box['text'] for box in text_new['singleBox']]
                        text = " ".join(text_new_list)
                        if type == 'title':
                            current_title = text
                            continue
                        vertices = item['expand_quad_location'] ### 边框顶点坐标
                        min_x = min(v['x'] for v in vertices)
                        min_y = min(v['y'] for v in vertices)
                        max_x = max(v['x'] for v in vertices)
                        max_y = max(v['y'] for v in vertices)
                        cropped_image = image.crop((min_x, min_y, max_x, max_y))
                        width, height = cropped_image.size
                        buffer = io.BytesIO()
                        cropped_image.save(buffer, format="PNG")
                        img_bytes = buffer.getvalue()
                        img_base64 = base64.b64encode(img_bytes).decode()
                        response = self.vl_1b(img_base64)
                        if self.merge_overset_flag:
                            response = merge_overset(response)
                        if response:
                            decode_res = self.extract_internvl_response(response, image_path=None, width=width, height=height)
                            vl_handtext_list = self.convert_vl_resp_to_single_box(decode_res, vertices) ### 转录为single_box格式
                            vl_handtext_list = self.add_topN_into_singlebox_list(vl_handtext_list) ### 添加topN结果

                            merged_list = self.merge_singlebox(chailu_singlebox_list, vl_handtext_list) ### 合并拆录和vl模型的结果
                            sorted_list = self.sorted_bbox(merged_list)
                            text_new_vl = [box['text'] for box in sorted_list if box['text'].strip() != "####"]
                            text_vl = " ".join(text_new_vl)
                            hand_text_list = [box for box in sorted_list if '##' in box['text']]

                        else:
                            text_vl = ""
                            hand_text_list = [box for box in chailu_singlebox_list if '##' in box['text']]
                            
                        all_hand_list.extend(hand_text_list)
                        output.append({
                                "img_url": img_url,  # img_url is not used in this context
                                "vertices": vertices,
                                "title": current_title,
                                "text": text,
                                "VL_response": response,
                                "VL_decode_res": decode_res,
                                "hand_text_list": hand_text_list,
                                "text_vl": text_vl,
                                "diff_vl": text_vl != text,
                            })
        return output, all_hand_list

    def visual(self, hand_text_list, save_path):
        image = Image.open(self.img_path)
        draw = ImageDraw.Draw(image)
        image2 = Image.new('RGB', image.size, (255, 255, 255))
        draw2 = ImageDraw.Draw(image2)
        if not hand_text_list:
            hand_text_list = self.extract_hand_text_list()

        for box in hand_text_list:
            x = box['x']
            y = box['y']
            w = box['width']
            h = box['height']
            draw.rectangle([x, y, x+w, y+h], outline="blue", width=2)
            draw2.rectangle([x, y, x+w, y+h], outline="blue", width=2)
            ans_text = box['text'].replace('##', '')
            ans_text = re.sub(r'\\handcheckmark\{(.*?)\}', r'\1', ans_text)
            ans_text = re.sub(r'\\underline\{(.*?)\}', r'\1', ans_text)
            ans_text = ans_text.replace('✔', '✓')
            ans_text = ans_text.replace('✘', '×')
            font = self.smart_font_size(ans_text, w, self.FontPath, min_fontsize=5)
            draw2.text((x,y), ans_text, fill="black", font=font)

        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image2 = np.array(image2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        out = np.hstack([image, image2])  # 横向拼接图像
        cv2.imwrite(save_path, out)
        return 


    def extract_hand_text_list(self):
        all_hand_list = []
        image = self.image
        ocr_res = self.ocr_res
        img_url = ocr_res['img_url']
        result = ocr_res['data']['result']
        output = []
        current_title = ""
        for idx, result_ in enumerate(result):
            result_data = result_['data']
            for data_i, data_ in enumerate(result_data):
                for item_i, item in enumerate(data_):
                    type = item['type']
                    text_new = item['text_new']
                    graph = item['graph'] if 'graph' in item else None
                    
                    if text_new['singleBox'] is not None:
                        # chailu_singlebox_list = self.add_topN_into_singlebox_list(text_new['singleBox']) ### 对拆录的手写体添加topN结果
                        chailu_singlebox_list = text_new['singleBox']
                        text_new_list = [box['text'] for box in text_new['singleBox']]
                        text = " ".join(text_new_list)
                        if type == 'title':
                            current_title = text
                            continue
                        vertices = item['expand_quad_location'] ### 边框顶点坐标
                        min_x = min(v['x'] for v in vertices)
                        min_y = min(v['y'] for v in vertices)
                        max_x = max(v['x'] for v in vertices)
                        max_y = max(v['y'] for v in vertices)
                        cropped_image = image.crop((min_x, min_y, max_x, max_y))
                        width, height = cropped_image.size
                        buffer = io.BytesIO()
                        cropped_image.save(buffer, format="PNG")
                        img_bytes = buffer.getvalue()
                        img_base64 = base64.b64encode(img_bytes).decode()
                        response = self.vl_1b(img_base64)
                        if self.merge_overset_flag:
                            response = merge_overset(response)
                        if response:
                            decode_res = self.extract_internvl_response(response, image_path=None, width=width, height=height)
                            vl_handtext_list = self.convert_vl_resp_to_single_box(decode_res, vertices) ### 转录为single_box格式
                            vl_handtext_list = self.add_topN_into_singlebox_list(vl_handtext_list) ### 添加topN结果

                            merged_list = self.merge_singlebox(chailu_singlebox_list, vl_handtext_list) ### 合并拆录和vl模型的结果
                            sorted_list = self.sorted_bbox(merged_list)
                            text_new_vl = [box['text'] for box in sorted_list if box['text'].strip() != "####"]
                            text_vl = " ".join(text_new_vl)
                            hand_text_list = [box for box in sorted_list if '##' in box['text']]

                        else:
                            text_vl = ""
                            hand_text_list = [box for box in chailu_singlebox_list if '##' in box['text']]
                        all_hand_list.extend(hand_text_list)
        return all_hand_list
                            

    def convert_vl_resp_to_single_box(self, decode_res, vertices):
        '''
        :param decode_res: dict, vl模型的解析结果
        :param vertices: list, 当前vl输出的题框坐标
        :return: 手写体list, 同试卷拆录的single_box list
        '''
        single_box = []
        x1 = min(v['x'] for v in vertices)
        y1 = min(v['y'] for v in vertices)
        for ans_box in decode_res['answer_list']:
            bbox = ans_box['bbox']
            ans_text = ans_box['answer']
            ### 如果ans_text包含latex标签，例如 \\handcheckmark{text}, 去除此类标签只保留text内容
            ans_text = re.sub(r'\\handcheckmark\{(.*?)\}', r'\1', ans_text)
            ans_text = re.sub(r'\\underline\{(.*?)\}', r'\1', ans_text)

            if bbox:
                min_x = min(v[0] for v in bbox)
                min_y = min(v[1] for v in bbox)
                max_x = max(v[0] for v in bbox)
                max_y = max(v[1] for v in bbox)
                single_box.append({
                    "height": max_y - min_y,
                    "is_print": False,
                    "text": f"##{ans_text}##",
                    "width": max_x - min_x,
                    "x": min_x + x1,
                    "y": min_y + y1
                  })
        return single_box

    def merge_singlebox(self, chailu_singlebox, vl_singlebox):
        '''
        :param chailu_singlebox: list, 拆录的单选框
        :param vl_singlebox: list, vl模型的单选框
        :return: 合并后的单选框list
        '''
        
        for box in vl_singlebox[:]:
            box1 = [box['x'], box['y'], box['x']+box['width'], box['y']+box['height']]
            for box_exist in chailu_singlebox:
                is_print = box_exist.get('is_print', "")
                if is_print:
                    continue
                box2 = [box_exist['x'], box_exist['y'], box_exist['x']+box_exist['width'], box_exist['y']+box_exist['height']]
                if self.judge_overlap(box1, box2, threshold=0.5):
                    vl_singlebox.remove(box)
                    break
        merged_singlebox = chailu_singlebox + vl_singlebox
        return merged_singlebox
    
    
    def remove_duplicates(self, single_box_list: list) -> list:
        """
        去掉重复的手写体框（手写体内容一致且坐标重叠的框）
        :param single_box_list: list, 手写体框列表
        :return: list, 去重后的手写体框列表
        """
        unique_boxes = []

        return
                    
    def sorted_bbox(self, single_box_list: list) -> list:
        """
        Sorts the single_box list based on the bounding box coordinates.
        """
        def get_center(box):
            bbox = [box['x'], box['y'], box['x']+box['width'], box['y']+box['height']]
            # bbox = item.get('bbox', [])
            ### center_x
            y_coords = [bbox[1], bbox[3]]
            center_y = sum(y_coords) / 2
            # center_x
            x_coords = [bbox[0], bbox[2]]
            center_x = sum(x_coords) / 2
            return center_y, center_x

        # y_threshold = 50  # y坐标差值阈值
        height_list = [box['height'] for box in single_box_list]
        if height_list:
            y_threshold = max(height_list) * 0.5  # 动态调整y_threshold为最大高度的50%


        items_with_coords = [(item, get_center(item)) for item in single_box_list]
         # 按y坐标粗略分组
        y_groups = {}
        for item, (center_y, center_x) in items_with_coords:
            # 找到最近的y坐标组
            found_group = False
            for group_y in y_groups.keys():
                if abs(group_y - center_y) < y_threshold:
                    y_groups[group_y].append((item, center_x))
                    found_group = True
                    break
            
            # 如果没有找到合适的组，创建一个新组
            if not found_group:
                y_groups[center_y] = [(item, center_x)]

        # 对每一行，按x坐标从左到右排序
        result = []
        # 按y坐标从上到下遍历每一行
        for group_y in sorted(y_groups.keys()):
            # 对当前行的项目按x坐标排序
            sorted_row = sorted(y_groups[group_y], key=lambda x: x[1])
            # 只保留项目对象，不要坐标
            result.extend([item for item, _ in sorted_row])

        return result
    
    
    def judge_overlap(self, box1, box2, threshold=0.5):
        """
        判断两个框是否重叠
        :param box1: dict, 第一个框
        :param box2: dict, 第二个框
        :return: bool, 如果重叠比例超过阈值则返回True，否则返回False
        """
        def compute_overlap(box1, box2):
            """
            计算两个矩形框的重叠面积及其占各自面积的比例
            参数:
            box1, box2: [x1, y1, x2, y2] 格式的矩形框，(x1,y1)为左上角坐标，(x2,y2)为右下角坐标
            返回:
            overlap_area: 重叠区域的面积
            ratio1: 重叠区域占box1的比例
            ratio2: 重叠区域占box2的比例
            """
            # 计算各自矩形的面积
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            # 计算重叠区域的坐标
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])
            
            # 检查是否存在重叠
            if x_right < x_left or y_bottom < y_top:
                return 0, 0, 0  # 没有重叠
            
            # 计算重叠面积
            overlap_area = (x_right - x_left) * (y_bottom - y_top)
            
            # 计算重叠区域占各自矩形的比例
            ratio1 = overlap_area / area1
            ratio2 = overlap_area / area2
            
            return overlap_area, ratio1, ratio2
        
        overlap_area, ratio1, ratio2 = compute_overlap(box1, box2)

        return overlap_area > 0 and (ratio1 > threshold or ratio2 > threshold)
    

    def add_topN_into_singlebox_list(self, single_box_list: list) -> list:
        '''
        对single_box_list中的每个框，调用topN接口
        :param single_box_list: list, 手写体框列表
        :return: list, 替换text后的手写体框列表
        '''
        image = self.image
        for box in single_box_list:
            if not box['is_print']:
                x = box['x']
                y = box['y']
                cropped_image = image.crop((x, y, x+box['width'], y+box['height']))
                cropped_image_np = np.array(cropped_image)
                ocr_topN = get_topN_result([cropped_image_np])
                try:
                    topN_text = ocr_topN['data']['result'][0] if 'data' in ocr_topN else ""
                except:
                    print(ocr_topN)
                    topN_text = ""

                ### 将topN结果添加到single_box中
                box.update({
                    "topN": topN_text
                })
        return single_box_list




def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge VL Chailu")
    parser.add_argument("--path", type=str, required=True, help="Path to the directory.")
    parser.add_argument("--merge_overset", action='store_true')
    args = parser.parse_args()
    path = args.path

    # path = '/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/jiaofu/db_data/0822-1/merge_vlocr'
    # path = '/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/jiaofu/db_data/0822-2/merge_vlocr'
    # ocr_dir = f'{path}/jiaozheng_ocr'
    ocr_path = f'{path}/ocr_supp.json'
    ocr_supp = json.load(open(ocr_path, 'r', encoding='utf-8'))
    img_dir = f'{path}/jiaozheng'
    temp_dir = f'{path}/temp'
    os.makedirs(temp_dir, exist_ok=True)

    visual_dir = f'{path}/visual'
    os.makedirs(visual_dir, exist_ok=True)

    outputs = []
    current_id = 0
    url_to_id = OrderedDict()
    for ocr_res in tqdm(ocr_supp):
        img_url = ocr_res['img_url']
        img_name = os.path.split(img_url)[-1]
        img_path = os.path.join(img_dir, img_name)

        if img_url and img_url not in url_to_id:
            url_to_id[img_url] = current_id
            current_id += 1
        
        temp_save_path = os.path.join(temp_dir, f'{img_name}_supp_merged.json')
        

        # input_json = os.path.join(ocr_dir, img_name)
        # img_path = os.path.join(img_dir, img_name.replace('.json', ''))
        # ocr_res = json.load(open(input_json, 'r', encoding='utf-8'))

        Merge_VL = Merge_VL_Chailu(img_path, ocr_res, merge_overset_flag=args.merge_overset)
        merged_res, hand_text_list = Merge_VL()

        Merge_VL.visual(hand_text_list, save_path=os.path.join(visual_dir, img_name))
        print(f"Visual results saved to {os.path.join(visual_dir, img_name)}")

        for item in merged_res:
            item["num"] = current_id

        with open(temp_save_path, 'w', encoding='utf-8') as f:
            json.dump(merged_res, f, ensure_ascii=False, indent=2)

        outputs += merged_res
    df = pd.DataFrame(outputs)
    output_csv = f'{path}/merged_ocr_supp_results.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f'Merged results saved to {output_csv}')

if __name__ == "__main__":
    main()