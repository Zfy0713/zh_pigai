from typing import List
import requests, os, sys, csv, json
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import uuid
import copy
from utils.utils import load_json, save_json, load_csv_2_dict, download_image, img_to_base64, base2img, remove_punctuation_and_special_chars
from utils.frame_utils import *
from utils.split_text import *
from utils.ocr_topN import get_topN_result
from utils.supp_detect import supp_detect
# from utils.merge_vl_overset import merge_overset
# from vl_ocr import VL_pigai_OCR
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_url', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    return parser.parse_args()


class ChailuOCR:
    def __init__(self, img_url, path, is_visual=True):
        self.img_url = img_url
        self.path = path
        self.img_name = os.path.split(img_url)[-1]

        if not any(self.img_name.lower().endswith(pic_format) for pic_format in ['jpg', 'png', 'jpeg']):
            self.img_name = self.img_name + '.jpg'

        self.save_path = f"{path}/jiaozheng/{self.img_name}"
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        if not os.path.exists(self.save_path):
            download_image(self.img_url, self.save_path)
        
        self.is_visual = is_visual ### 是否作可视化 (todo...)
        self.FontPath='/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/utils/SimHei.ttf'

        self.ocr_res = self.chailu_api()
        self.ocr_res_with_topN = self.add_topN(self.ocr_res) ### 给拆录结果singlebox添加手写体topN

        self.supp_res = self.chailu_supp_api() ### 作答补充检测结果

    def chailu_api(self, img_url=None):
        if img_url is None:
            img_url = self.img_url
        else:
            img_url = img_url
        
        ocr_json = f'{self.path}/jiaozheng_ocr'
        os.makedirs(ocr_json, exist_ok=True)
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(f"图片{self.save_path}不存在")

        base64_data = img_to_base64(self.save_path)
        requestId = uuid.uuid4()
        url = f"http://mlops-infer.tal.com/appset/ocr-fusion/?requestId={requestId}"      
        body_params = {"image_base64": base64_data}
        headers = {'content-type': 'application/json'}
        response = requests.post(url, json=body_params, headers=headers, timeout=20)
        res = response.text
        if not isinstance(res, dict):
            res = json.loads(res)
            res.update(dict(
                img_url=img_url
            ))
            ### 保存单张图片ocr结果
        save_json(f"{ocr_json}/{self.img_name}.json", res)
        print(f"试卷拆录结果保存至 {ocr_json}/{self.img_name}.json")

        return res

    def chailu_supp_api(self):
        supp_dir = f"{self.path}/jiaozheng_supp"
        os.makedirs(supp_dir, exist_ok=True)
        supp_json = f"{supp_dir}/{self.img_name}.json"
        supp_res = supp_detect(self.save_path)
        save_json(supp_json, supp_res)
        print(f"作答区域补充检测结果保存至 {supp_json}")

        return supp_res
    
    def visual(self, ocr_res:dict=None, supp_res:dict=None):
        '''
        可视化文本框/作答补充检测框
        '''
        if not ocr_res:
            ocr_res = self.ocr_res_with_topN
        if not supp_res:
            supp_res = self.supp_res
        
        img_path = self.save_path
        image = Image.open(img_path)
        draw = ImageDraw.Draw(image)
        image2 = Image.new('RGB', image.size, (255, 255, 255))
        draw2 = ImageDraw.Draw(image2)

        ### 试卷拆录框
        result = ocr_res['data']['result'] ### 拆录结果
        for idx, result_ in enumerate(result):
            result_data = result_['data']
            for data_ in result_data:
                for item_i, item in enumerate(data_):
                    text_new = item['text_new']
                    if text_new['singleBox'] is not None:
                        for box in text_new['singleBox']:
                            x = box['x']
                            y = box['y']
                            w = box['width']
                            h = box['height']
                            draw.rectangle([x, y, x+w, y+h], outline="blue", width=2)
        
        ### 作答补充检测框
        frames_list = supp_res['data']['frames'] ### 补充做答结果 
        # frames_list = [frame for frame in frames_list if 'type' in frame and frame['type']== "pinyin"] ### 只保留拼音题框
        for frame in frames_list:
            type = frame.get('type', '')
            min_x = min([v['x'] for v in frame['frame']])
            max_x = max([v['x'] for v in frame['frame']])
            min_y = min([v['y'] for v in frame['frame']])
            max_y = max([v['y'] for v in frame['frame']])
            box2 = [min_x, min_y, max_x, max_y]
            draw2.rectangle(box2, outline="green", width=2)
            font = ImageFont.truetype(self.FontPath, size = 15)
            draw2.text((min_x, min_y), text = type, fill='black', font=font)

        img_name = os.path.split(img_path)[-1]
        os.makedirs(os.path.join(path, 'visual'), exist_ok=True)
        save_visual_path = os.path.join(path, 'visual', img_name)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image2 = np.array(image2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        out = np.hstack([image, image2])  # 横向拼接图像
        cv2.imwrite(save_visual_path, out)


    def process_supp_pinyin(self, ocr_res: dict=None, supp_res: dict=None) -> list:
        '''
        return: 作答补充检测 pinyin_box_list
        '''
        if not ocr_res:
            ocr_res = self.ocr_res_with_topN
        if not supp_res:
            supp_res = self.supp_res
        
        output = []
        image_path = self.save_path
        if not os.path.exists(image_path):
            print(f'Image path {image_path} does not exist!')
            return output
        if len(supp_res['data']['frames'])== 0 or supp_res['code'] != 20000:
            return output
        
        frames_list = supp_res['data']['frames'] ### 补充做答结果 
        frames_list = [frame for frame in frames_list if 'type' in frame and frame['type']== "pinyin"] ### 只保留拼音题框
        frames_list = sort_frame(frames_list) ### 对补充做答区域排序
        frames_list_copy = copy.deepcopy(frames_list)

        flatten_singlebox = [] ### 试卷拆录singlebox信息
        result = ocr_res['data']['result'] ### 拆录结果
        for idx, result_ in enumerate(result):
            result_data = result_['data']
            for data_ in result_data:
                for item_i, item in enumerate(data_):
                    text_new = item['text_new']
                    if text_new['singleBox'] is not None:
                        for box in text_new['singleBox']:
                            flatten_singlebox.append(box)

        for j, frame in enumerate(frames_list_copy):

            xs = [v['x'] for v in frame['frame']]
            ys = [v['y'] for v in frame['frame']]
            hand_text = text = ""
            box2 = [min(xs), min(ys), max(xs), max(ys)]
            for i, singlebox_info in enumerate(flatten_singlebox):
                box1 = [singlebox_info['x'], singlebox_info['y'], singlebox_info['x'] + singlebox_info['width'], singlebox_info['y'] + singlebox_info['height']]
                                            
                if judge_overlap(box1, box2, threshold=0.6): ### 判断singlebox是否与做答补充检测区域重叠
                    if not singlebox_info['is_print']:
                        hand_text = singlebox_info['text']
                        hand_bbox = box1
                        topN_res = singlebox_info.get('topN', [])
                    else:
                        text += singlebox_info['text']
                    
            output.append({
                "hand_text": hand_text if hand_text else "",
                "text": text,
                "hand_bbox": hand_bbox if hand_text else [],
                "topN": topN_res if hand_text else []
            })

        return output

    def process_supp(self, ocr_res: dict=None, supp_res: dict=None):
        '''
        return: 融合作答补充检测后的single_box list
        '''
        if not ocr_res:
            ocr_res = self.ocr_res_with_topN
        if not supp_res:
            supp_res = self.supp_res

        image_path = self.save_path
        if not os.path.exists(image_path):
            print(f'Image path {image_path} does not exist!')
            return ocr_res
        
        if supp_res['code'] != 20000:
            supp_res = {'data': {'frames': []}}
        
        frames_list = supp_res['data']['frames'] ### 补充做答结果
        frames_list = [frame for frame in frames_list if 'type' in frame and frame['type']!= "pinyin"] ### 过滤掉拼音题框
        frames_list = sort_frame(frames_list) ### 对补充做答区域排序
        frames_list_copy = copy.deepcopy(frames_list)

        result = ocr_res['data']['result'] ### 拆录结果
        for idx, result_ in enumerate(result):
            result_data = result_['data']
            for data_ in result_data:
                for item_i, item in enumerate(data_):                    
                    text_new = item['text_new']
                    if text_new['singleBox'] is not None:
                        insert_len = 0
                        for i, singlebox_info in enumerate(text_new['singleBox'][:]):
                            
                            for j, frame in enumerate(frames_list_copy[:]):
                                supp_text = extract_text(singlebox_info, frame['frame'], expansion_amount = 0.8)
                                if len(supp_text) == 0:
                                    continue
                                
                                print(i,'  ---  ', i+insert_len+1,  supp_text, '  ---  ', singlebox_info['text'])
                                new_singlebox = convert_to_singlebox(frame['frame'], supp_text)
                                text_new['singleBox'].insert(i+insert_len+1, new_singlebox)
                                insert_len += 1
                                frames_list_copy.remove(frame) ## 找到后删除当前框坐标

        return ocr_res

    def chaiti(self, ocr_res: dict=None, save_path=None, is_split=True, split_num=20) -> list:
        if not ocr_res:
            ocr_res = self.process_supp() ### 融合作答补充检测后的拆录结果
            # ocr_res = self.ocr_res_with_topN
            
        if not save_path:
            save_dir = os.path.join(self.path, 'chaiti')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, self.img_name+'.jsonl')
        outputs = []

        result = ocr_res['data']['result']
        current_title = ""
        for idx, result_ in enumerate(result):
            result_data = result_['data']
            for data_ in result_data:
                for item in data_:
                    type = item['type']
                    text_new = item['text_new']
                    if text_new['singleBox'] is not None:

                        text_new_list = [box['text'] for box in text_new['singleBox']]
                        text = " ".join(text_new_list)
                        if type == 'title':
                            current_title = text
                            continue
                        vertices = item['expand_quad_location'] ### 边框顶点坐标

                        cnts = count_kongshu(text)
                        hand_text_list = [box for box in text_new['singleBox'] if '##' in box['text']]
                        if is_split and cnts>split_num:
                            maxK = split_num
                            split_res = split_ocr(text, max_kongshu = maxK)
                            for part in split_res:
                                hand_len = count_kongshu(part) 
                                temp_hand_text_list = hand_text_list[:hand_len]
                                hand_text_list = hand_text_list[hand_len:]
                                text = current_title + " " + part

                                # outputs.append([res['img_url'], str(vertices), current_title, text, temp_topN, temp_hand_text_list])
                                outputs.append({
                                    "img_url": ocr_res['img_url'],
                                    "vertices": vertices,
                                    "title": current_title,
                                    "text": text,
                                    "hand_text_list": temp_hand_text_list
                                })
                        else:
                            text = current_title + " " + text 
                            outputs.append({
                                "img_url": ocr_res['img_url'],
                                "vertices": vertices,
                                "title": current_title,
                                "text": text,
                                "hand_text_list": hand_text_list
                            })
        
        with open(save_path, 'w', encoding='utf-8') as f:
            for out in outputs:
                f.write(json.dumps(out, ensure_ascii=False) + '\n')
        print(f"拆题结果保存至 {save_path}")

        return outputs

    def add_topN(self, ocr_res: dict=None) -> dict:
        '''
        给拆录结果singlebox添加手写体topN
        singlebox: {'x','y','width','height','is_print','text'}
        '''
        if not ocr_res:
            ocr_res = self.ocr_res
        image = Image.open(self.save_path)
        result = ocr_res['data']['result']
        for idx, result_ in enumerate(result):
            result_data = result_['data']
            for data_ in result_data:
                for item in data_:
                    text_new = item['text_new']
                    if text_new['singleBox'] is not None:
                        for box in text_new['singleBox']:
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
                                box.update({
                                    "topN": topN_text
                                })
        return ocr_res



if __name__ == "__main__":
    # args = get_args()
    # chailu_api(args)
    # url = 'https://ss-prod-genie.oss-cn-beijing.aliyuncs.com/correct_pipeline/processed_image/2025-09-07/ecb6c929-4fc2-4108-9a0d-9f729b74d2fe.jpg'
    url = 'https://ss-prod-genie.oss-cn-beijing-internal.aliyuncs.com/correct_pipeline/processed_image/2025-09-27/85bf7b88-141d-430c-98d0-75ba641c26cd.jpg'
    path = '/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_pipeline/ocr_pipeline/test_dir'
    ChailuInstance = ChailuOCR(img_url=url, path=path)
    pinyin_output = ChailuInstance.process_supp_pinyin()
    ChailuInstance.visual()
    ChailuInstance.chaiti()
    # print(pinyin_output)


