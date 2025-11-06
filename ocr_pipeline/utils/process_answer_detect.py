import json, csv, os, sys
import copy
from time import sleep
from PIL import Image, ImageDraw, ImageFont
# from get_ocr import get_ocr_by_box, get_ocr_by_box_with_title
# from get_ocr_topN import get_ocr_with_topN

EXPANSION_SCALE = 0.8 ### 定义做答区域左右扩展比例，按照‘字符像素宽度*EXPANSION_SCALE’大小扩展
FONT_PATH='/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/utils/SimHei.ttf'

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data
def save_json(save_path, data):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def sort_frame(frames):
    '''
    将frame根据坐标顺序排序后返回
    '''

    # 找到左上角顶点坐标
    def top_left_corner(frame):
        min_x = min(point['x'] for point in frame)
        min_y = min(point['y'] for point in frame)
        return min_x, min_y

    ### 根据左上角坐标，先对x排序，在对y排序
    sorted_frames = sorted(frames, key=lambda f: (top_left_corner(f["frame"])[0], top_left_corner(f["frame"])[1]))
    return sorted_frames

# 检查点是否在矩形范围内
def is_point_in_rectangle(x_min, y_min, x_max, y_max, x0, y0):
    # 判断点是否在矩形的x范围内
    if x_min <= x0 <= x_max:
        # 判断点是否在矩形的y范围内
        if y_min <= y0 <= y_max:
            return True
    return False

# 检查点是否在多边形内的函数
def is_point_in_polygon(px, py, polygon):
    flag = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[j]
        if ((y1 > py) != (y2 > py)) and (px < x1 + (x2 - x1) * (py - y1) / (y2 - y1)):
            flag = not flag
        j = i
    return flag

def convert_to_singlebox(answer_frame, extract_text):
    '''
    将监测到的补充做答区域和文本转换成singlebox相同格式
    '''
    x_min = 99999
    y_min = 99999
    x_max = -99999
    y_max = -99999
    for i in answer_frame:
        x=int(i['x'])
        y=int(i['y'])
        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x, x_max)
        y_max = max(y, y_max)
    out = {
        "height": y_max - y_min,
        "is_print": False,
        "text": f"##{extract_text}##",
        "width": x_max - x_min,
        "x": x_min,
        "y": y_min,
        "source": "supp_detect"
    }
    return out

def extract_text(ocr_singlebox, answer_frame, expansion_amount = 2):
    '''
    ocr_singlebox: 试卷拆录结果中的single_box字段。 e.g.{"height": 30, "is_print": True, "text": "似乎(shì sì) 扇动(shān shàn) 头痛(téng tòng)", "width": 338, "x": 58, "y": 214}
    answer_frame: 补充检测得到的区域坐标。 e.g.[{"x": 356,"y": 231},{"x": 356,"y": 213},{"x": 387,"y": 213}, {"x": 387, "y": 232}]
    expansion_amount: 扩宽一定的做答区域,只放宽左右区域
    '''
    original_polygon = [(v['x'], v['y']) for v in answer_frame]
    x_min = 99999
    y_min = 99999
    x_max = -99999
    y_max = -99999
    for i in answer_frame:
        x=int(i['x'])
        y=int(i['y'])
        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x, x_max)
        y_max = max(y, y_max)
    
    # 定义文本区域的矩形坐标
    text_x = ocr_singlebox["x"]
    text_y = ocr_singlebox["y"]
    text_width = ocr_singlebox["width"]
    text_height = ocr_singlebox["height"]

    # 计算文本字符串的每个字符占据的宽度
    char_width = text_width / len(ocr_singlebox["text"])

    ## 放宽矩形左右区域
    # x_min -= expansion_amount * char_width*1.5
    x_min -= expansion_amount * char_width*1.5
    x_max += expansion_amount * char_width

    extracted_text = ""

    # 循环检查每个字符的中心点是否在多边形内
    positions = []
    for i, char in enumerate(ocr_singlebox["text"]):
        # 计算每个字符的中心点
        char_center_x = text_x + char_width * (i + 0.5)
        positions.append({
            'ocr': char, 'x_pos': char_center_x
        })
        char_center_y = text_y + text_height / 2
        
        # 如果字符中心点在多边形内，添加到结果中
        # if is_point_in_polygon(char_center_x, char_center_y, polygon):
        #     extracted_text += char

        # 如果字符中心点在扩展矩形内，添加到结果中
        if is_point_in_rectangle(x_min, y_min, x_max, y_max, char_center_x, char_center_y):
            extracted_text += char
    return extracted_text

def process_supp(ocr_origin, supp_dir):
    '''
    ocr_origin: 原始拆录结果
    supp_dir: 做答补充检测结果目录
    output: ocr_origin_new, 将补充检测结果合并至拆录结果中
    '''
    ocr_new = copy.deepcopy(ocr_origin)
    for num, res in enumerate(ocr_new):
        if res['code'] != 20000:
            continue
        
        img_url = res['img_url']  ### e.g. 003184.jpg
        image_name = os.path.split(img_url)[-1]

        supp_json = f"{supp_dir}/{image_name}.json"
        if not os.path.exists(supp_json):
            print(f'answer supplement file path {supp_json} does not exist!')
            continue
        supp_ocr = load_json(supp_json)
        if not isinstance(supp_ocr, dict):
            supp_ocr = json.loads(supp_ocr)
        if supp_ocr['code'] != 20000:
            continue
        if len(supp_ocr['data']['frames'])== 0 or supp_ocr['code'] != 20000:
            continue
        frames_list = supp_ocr['data']['frames'] ### 补充做答结果

        ### Update
        frames_list = [frame for frame in frames_list if 'type' in frame and frame['type']!= "pinyin"] ### 过滤掉没有拼音题框
        frames_list = sort_frame(frames_list) ### 对补充做答区域排序
        frames_list_copy = copy.deepcopy(frames_list)

        result = res['data']['result'] ### 拆录结果
        
        for idx, result_ in enumerate(result):
            result_data = result_['data']
            common_ocr_results = result_['common_ocr_results']
            insert_len = 0
            for i, common_ocr_frame in enumerate(common_ocr_results[:]):
                for j, frame in enumerate(frames_list[:]):
                    supp_text = extract_text(common_ocr_frame, frame['frame'], expansion_amount = EXPANSION_SCALE)
                    if len(supp_text) == 0:
                        continue
                    new_singlebox = convert_to_singlebox(frame['frame'], supp_text)
                    common_ocr_results.insert(i+insert_len+1, new_singlebox)
                    insert_len += 1
                    frames_list.remove(frame)

            for data_ in result_data:
                for item_i, item in enumerate(data_):
                    
                    text_new = item['text_new']
                    if text_new['singleBox'] is not None:
                        # text_new_list = [box['text'] for box in text_new['singleBox']]
                        # text = " ".join(text_new_list)
                        # singlebox_info_new = singlebox_info[:]
                        insert_len = 0
                        for i, singlebox_info in enumerate(text_new['singleBox'][:]):
                            
                            for j, frame in enumerate(frames_list_copy[:]):
                                supp_text = extract_text(singlebox_info, frame['frame'], expansion_amount = EXPANSION_SCALE)
                                if len(supp_text) == 0:
                                    continue
                                
                                print(i,'  ---  ', i+insert_len+1,  supp_text, '  ---  ', singlebox_info['text'])
                                new_singlebox = convert_to_singlebox(frame['frame'], supp_text)
                                text_new['singleBox'].insert(i+insert_len+1, new_singlebox)
                                insert_len += 1
                                frames_list_copy.remove(frame) ## 找到后删除当前框坐标
    return ocr_new
       

if __name__ == "__main__":

    ### visualize supplement area
    supp_path='/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/0325data/jiaozheng_supp/b905a9e9-4942-43da-859d-c40b05cda19e.jpg.json'
    img_path='/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/0325data/rag-hegang/jiaozheng/b905a9e9-4942-43da-859d-c40b05cda19e.jpg'
    ocr_origin_path='/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/0325data/ocr_origin.json'
    ocr_origin = load_json(ocr_origin_path)
    supp_ocr = load_json(supp_path)
    if not isinstance(supp_ocr, dict):
        supp_ocr = json.loads(supp_ocr)
    
    
    # frames_list = supp_ocr['data']['frames'] ### 补充做答结果
    # frames_list = sort_frame(frames_list) ### 对补充做答区域排序
    # image = Image.open(img_path)
    # draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype(FONT_PATH, 20)
    # for frame in frames_list:
    #     # frame = {'frame': [{'x': 866, 'y': 756}, {'x': 866, 'y': 720}, {'x': 924, 'y': 720}, {'x': 924, 'y': 756}], 'type': 'select_mark'}
    #     original_polygon = [(v['x'], v['y']) for v in frame['frame']]
    #     # points = [(v['x'], v['y']) for v in frame['frame']]
        
    #     polygon = []
    #     for (x, y) in original_polygon:
    #         if x == original_polygon[0][0]:  # 左边的顶点
    #             x -= EXPANSION_SCALE
    #         elif x == original_polygon[2][0]:  # 右边的顶点
    #             x += EXPANSION_SCALE
    #         polygon.append((x, y))
    #     draw.polygon(polygon, outline='red', width=2)

    # for num, res in enumerate(ocr_origin):
    #     image_name='b905a9e9-4942-43da-859d-c40b05cda19e.jpg'
    #     if image_name not in res['img_url']:
    #         continue
    #     result = res['data']['result']
    #     for idx, result_ in enumerate(result):
    #         result_data = result_['data']
    #         for data_ in result_data:
    #             for item_i, item in enumerate(data_):
    #                 text_new = item['text_new']
    #                 if text_new['singleBox'] is not None:
    #                     for i, singlebox_info in enumerate(text_new['singleBox'][:]):
    #                             char_width = singlebox_info['width'] / len(singlebox_info["text"])
    #                             draw.rectangle([singlebox_info['x'], singlebox_info['y'], singlebox_info['x']+singlebox_info['width'], singlebox_info['y']+singlebox_info['height']], outline='blue', width = 4)
    #                             for j in range(1, len(singlebox_info["text"])):  # 从第1个字符开始
    #                                 x_pos = singlebox_info['x'] + j * char_width
    #                                 draw.line(
    #                                     [(x_pos, singlebox_info['y']), (x_pos, singlebox_info['y'] + singlebox_info['height'])],
    #                                     fill='green',
    #                                     width=2
    #                                 )

    # save_dir = '/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/0325data/jiaozheng_supp_vis'
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, os.path.split(img_path)[-1])
    # image.save(save_path)

    supp_dir='/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/0325data/jiaozheng_supp'
    ocr_origin_path='/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/data/0325data/jiaozheng_ocr/b905a9e9-4942-43da-859d-c40b05cda19e.jpg.json'
    ocr_origin = load_json(ocr_origin_path)
    ocr_new = process_supp([ocr_origin], supp_dir)
    # sleep(10)