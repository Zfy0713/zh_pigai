import json
import os
import re
import io
import base64
from tqdm import tqdm
import requests
from PIL import Image
from bs4 import BeautifulSoup
import html
from ocr_topN import get_topN_result


def load_json(file_path):
    """Load a JSON file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

class SmartSearchProcessor:
    def __init__(self, input_json_path):
        self.data = load_json(input_json_path)
    
    def convert_url_to_base64(self, img_url):
        """
        img_url: str, 图片的网络地址
        返回图片的 base64 字符串
        """
        response = requests.get(img_url)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64

    def crop_image_url_to_base64(self, img_url, crop_box):
        """
        img_url: str, 图片的网络地址
        crop_box: tuple, (left, upper, right, lower) 裁剪框坐标
        返回裁剪图的 base64 字符串
        """
        # 下载图片内容
        response = requests.get(img_url)
        response.raise_for_status()  # 若下载失败抛出异常

        img = Image.open(io.BytesIO(response.content))
        # img_gray = img.convert('L')  # 转为灰度图
        # print(img.size)
        # 裁剪图片
        cropped_img = img.crop(crop_box)
        # 保存至内存并转为 base64
        buffer = io.BytesIO()
        cropped_img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return img_base64

    def extract_answer_text(self, answer: list, img_path: str) -> str:
        """
        Extracts the text from an answer value.
        """
        ht_answer_tile_converted = []
        for ans in answer:
            ans_text = ans.get('text', '').strip()
            ans_text = html.unescape(ans_text)
            soup = BeautifulSoup(ans_text, 'html.parser')
            ans_text = soup.get_text()
            bbox = ans.get('bbox', [])
            tmp = {
                "ans_text": ans_text,
                "bbox": bbox
            }
            # ans_url = ans.get('url', None) ### 答案截图
            if ans_text == "圈选题" or ans_text == "<p>圈选题</p>":
                if img_path is None:
                    print("圈选题 -- No image path provided for cropping. Use ocrgin answer text instead.")
                    # ht_answer_tile_converted.append(ans_text)
                    # ht_answer_tile_converted.append(tmp)
                else:
                    bbox = ans.get('bbox', [])
                    [x1,y1,x2,y2,x3,y3,x4,y4] = bbox
                    min_x = min(x1, x2, x3, x4)
                    min_y = min(y1, y2, y3, y4)
                    max_x = max(x1, x2, x3, x4)
                    max_y = max(y1, y2, y3, y4)
                    crop_box = (min_x, min_y, max_x, max_y)
                    img_base64 = self.crop_image_url_to_base64(img_path, crop_box)
                    # comedu_ocr_res = comedu_ocr(img_base64)
                    topN_res = get_topN_result([img_base64])
                    quanxuan_ocr = topN_res['data']['result'][0]['results'][0]['text'].replace(' ','')
                    # ht_answer_tile_converted.append(quanxuan_ocr)
                    tmp['ans_text'] = quanxuan_ocr
            elif ans_text.startswith('<img alt'): ### 回填答案为截图
                soup = BeautifulSoup(ans_text, 'html.parser')
                img_tag = soup.find('img')
                if img_tag and img_tag.has_attr('src'):
                    src_url = img_tag['src']
                    img_base64 = self.convert_url_to_base64(src_url)
                    topN_res = get_topN_result([img_base64])
                    ocr_text = topN_res['data']['result'][0]['results'][0]['text'].replace(' ','')
                    # html_decoded = html.unescape(ocr_text)
                    # ht_answer_tile_converted.append(ocr_text)
                    tmp['ans_text'] = ocr_text
            elif '__' in ans_text or '~~' in ans_text or ans_text == '/': ## 画线/操作题
                bbox = ans.get('bbox', [])
                [x1,y1,x2,y2,x3,y3,x4,y4] = bbox
                min_x = min(x1, x2, x3, x4)
                min_y = min(y1, y2, y3, y4)
                max_x = max(x1, x2, x3, x4)
                max_y = max(y1, y2, y3, y4)
                crop_box = (min_x, min_y, max_x, max_y)
                img_base64 = self.crop_image_url_to_base64(img_path, crop_box)
                # comedu_ocr_res = comedu_ocr(img_base64)
                topN_res = get_topN_result([img_base64])
                quanxuan_ocr = topN_res['data']['result'][0]['results'][0]['text'].replace(' ','')
                tmp['ans_text'] = quanxuan_ocr

            tmp['ans_text'] = html.unescape(tmp['ans_text'])

            
            ht_answer_tile_converted.append(tmp)
        return ht_answer_tile_converted

    def sorted_bbox(self, items: list) -> list:
        """
        Sorts the ht_answer_tile_converted list based on the bounding box coordinates.
        """
        def get_center(item):
            bbox = item.get('bbox', [])
            ### center_x
            y_coords = [bbox[1], bbox[3], bbox[5], bbox[7]]
            center_y = sum(y_coords) / 4
            # center_x
            x_coords = [bbox[0], bbox[2], bbox[4], bbox[6]]
            center_x = sum(x_coords) / 4
            return center_y, center_x

        # y_threshold = 50  # y坐标差值阈值
        # Todo: dynamic adjustment of y_threshold based on the items 
        height_list = []
        for item in items:
            bbox = item.get('bbox', [])
            if bbox:
                height = max(bbox[1], bbox[3], bbox[5], bbox[7]) - min(bbox[1], bbox[3], bbox[5], bbox[7])
                height_list.append(height)
        if height_list:
            y_threshold = max(height_list) * 0.5


        items_with_coords = [(item, get_center(item)) for item in items]
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
            

    def process(self):
        """Process the loaded data and summary questions and answers."""
        ans_url = self.data.get('ans_img', None)
        search_page = self.data.get('search_page', None)
        if not search_page:
            print("Error: No search page information found in the input JSON.")
            return {}
            
        try:
            ext_info = search_page['ext']['page_info']
            book_id = ext_info.get('book_id', "")
            page_num = ext_info.get('page_num', "")
        except Exception as e:
            ext_info = {}
            print(f"Error retrieving book_id and page_num: {e}")
        
        img_path = search_page.get('img_path', None) ### 题库图
        blocks = search_page.get('blocks', [])
        ques_list = []
        for block in blocks:
            text = block.get('text', '').strip()
            # x='3.下列字形完全正确的一组是()。(3分)#A.偶而母鸡玄机#B.格楼庆典暂时#C.药丸厘米车轴#D.委曲陌生门逢'
            # if text != x:
            #     continue
            ht_answer_tile_converted = []
            type_tile = []
            single_list = block.get('single_list', [])
            for item in single_list:
                
                answer = item.get('answer', [])
                type_list = item.get('type_list', []) ### 题目类型列表
                ht_answer_tile_converted = self.extract_answer_text(answer, img_path)
                type_list.extend(type_tile) ### 扩展题目类型列表
                _single_list = item.get('single_list', []) ### 次级列表答案
                for __single_box in _single_list:
                    _type_list = __single_box.get('type_list', [])
                    _extend = __single_box.get('extend', {})
                    
                    answer = __single_box.get('answer', [])
                    ht_answer_tile_converted.extend(self.extract_answer_text(answer, img_path))
                    type_list.extend(_type_list)
                    is_open = _extend['openanswer'] if 'openanswer' in _extend else "unknown"
            
            # 排序 ht_answer_tile_converted
            ht_answer_tile_converted = self.sorted_bbox(ht_answer_tile_converted)
            
            ques_list.append({
                'text': text,
                'ht_answer_tile_converted': ht_answer_tile_converted,
                "type_list": type_list,
            })
        

        processed_page = {
            'book_id': book_id,
            'page_num': page_num,
            'img_path_tiku': img_path,
            'extracted_ques_list': ques_list,
        }

        return processed_page
        


class SmartSearchProcessor_new(SmartSearchProcessor):
    def __init__(self, input_json_path):
        super().__init__(input_json_path)
    
    def process(self):
        '''同时保存大题和小题两个拆分信息'''
        search_page = self.data.get('search_page', None)
        if not search_page:
            print("Error: No search page information found in the input JSON.")
            return {}
            
        try:
            ext_info = search_page['ext']['page_info']
            book_id = ext_info.get('book_id', "")
            page_num = ext_info.get('page_num', "")
        except Exception as e:
            ext_info = {}
            print(f"Error retrieving book_id and page_num: {e}")
        
        img_path = search_page.get('img_path', None) ### 题库图
        blocks = search_page.get('blocks', [])
        main_question_list = []
        sub_question_list = []
        
        for block in blocks:
            # main_text = block.get('text', '').strip()
            # x='3.下列字形完全正确的一组是()。(3分)#A.偶而母鸡玄机#B.格楼庆典暂时#C.药丸厘米车轴#D.委曲陌生门逢'
            # if text != x:
            #     continue
            main_answer_tile_converted = []
            # main_type_list = block.get('type_list', [])
            
            single_list = block.get('single_list', [])
            
            for item in single_list:
                main_text = item.get('text', '').strip()
                answer = item.get('answer', [])
                answer_extracted_list = self.extract_answer_text(answer, img_path)
                main_answer_tile_converted.extend(answer_extracted_list)
                main_type_list = item.get('type_list', []) ### 题目类型
                _single_list = item.get('single_list', []) ### 次级列表答案
                if not _single_list:
                    print("Error: No sub_question list found in the input JSON.")
                    
                
                for sub_single_box in _single_list:
                    sub_type_list = sub_single_box.get('type_list', [])
                    _extend = sub_single_box.get('extend', {})
                    sub_text = sub_single_box.get('text', '').strip()
                    answer = sub_single_box.get('answer', [])
                    answer_extracted_list = self.extract_answer_text(answer, img_path)
                    answer_sorted_list = self.sorted_bbox(answer_extracted_list)

                    sub_answer_tile_converted = answer_sorted_list
                    main_answer_tile_converted.extend(answer_sorted_list)
                    main_type_list.append(sub_type_list)
                    is_open = _extend['openanswer'] if 'openanswer' in _extend else "unknown"
            
                    # 排序 ht_answer_tile_converted
                    sub_question_list.append({
                        'text': sub_text,
                        'ht_answer_tile_converted': sub_answer_tile_converted,
                        "type_list": sub_type_list,
                        "level": "sub_question"
                    })

                main_question_list.append({
                    'text': main_text,
                    'ht_answer_tile_converted': self.sorted_bbox(main_answer_tile_converted),
                    "type_list": main_type_list,
                    "level": "main_question"
                })
        

        processed_page = {
            'book_id': book_id,
            'page_num': page_num,
            'img_path_tiku': img_path,
            'main_question_list': main_question_list,
            'sub_question_list': sub_question_list,
        }

        return processed_page

    def save_to_jsonl(self, processed_page, output_jsonl_path=None):
        '''处理processed_page格式，按题维度保存为jsonl格式'''
        outputs = []
        book_id = processed_page.get('book_id', "")
        page_num = processed_page.get('page_num', "")
        img_path = processed_page.get('img_path_tiku', "")
        main_question_list = processed_page.get('main_question_list', [])
        sub_question_list = processed_page.get('sub_question_list', [])
        for main_question in main_question_list:
            main_text = main_question.get('text', '')
            main_answer_tile_converted = main_question.get('ht_answer_tile_converted', [])
            main_type_list = main_question.get('type_list', [])
            level = main_question.get('level', '')
            outputs.append({
                'book_id': book_id,
                'page_num': page_num,
                'img_path_tiku': img_path,
                'question': main_text,
                'answer': main_answer_tile_converted,
                'type': main_type_list,
                'level': level
            })
        main_question_num = len(outputs)
        for sub_question in sub_question_list:
            sub_text = sub_question.get('text', '')
            sub_answer_tile_converted = sub_question.get('ht_answer_tile_converted', [])
            sub_type_list = sub_question.get('type_list', [])
            level = sub_question.get('level', '')
            outputs.append({
                'book_id': book_id,
                'page_num': page_num,
                'img_path_tiku': img_path,
                'question': sub_text,
                'answer': sub_answer_tile_converted,
                'type': sub_type_list,
                'level': level
            })
        sub_question_num = len(outputs) - main_question_num

        print(f"total {main_question_num} main questions, {sub_question_num} sub questions, total {main_question_num + sub_question_num} questions.")  
        if output_jsonl_path:
            with open(output_jsonl_path, 'w', encoding='utf-8') as f:
                for item in outputs:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Saved to {output_jsonl_path}")
        return outputs
        

def main(args):
    input_path = args.search_json_path
    save_path = args.processed_save_path

    
    processor = SmartSearchProcessor_new(input_path)
    processed_data = processor.process()
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(f"Processed and saved: {save_path}")

    # outputs = []
    # for filename in tqdm(os.listdir(input_dir)):
    #     if filename.endswith('.json'):
    #         input_json_path = os.path.join(input_dir, filename)
    #         processor = SmartSearchProcessor_new(input_json_path)
    #         processed_data = processor.process()
    #         outputs += processor.save_to_jsonl(processed_data, output_jsonl_path=None)

    #         output_json_path = os.path.join(output_dir, filename)
    #         # if os.path.exists(output_json_path):
    #         #     continue
    #         with open(output_json_path, 'w', encoding='utf-8') as f:
    #             json.dump(processed_data, f, ensure_ascii=False, indent=2)
    #         print(f"Processed and saved: {output_json_path}")

    # output_jsonl_path = os.path.join(path, 'smart_match_ruku_all.jsonl')
    # with open(output_jsonl_path, 'w', encoding='utf-8') as f:
    #     for item in outputs:
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')
    # print(f"Saved to {output_jsonl_path}, total {len(outputs)} questions.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--search_json_path', type=str)
    parser.add_argument('--processed_save_path', type=str)
    args = parser.parse_args()
    main(args)