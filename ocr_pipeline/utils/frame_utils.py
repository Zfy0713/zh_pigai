
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

def judge_overlap(box1, box2, threshold=0.5):
    """
    判断两个矩形框是否重叠超过指定阈值
    
    参数:
    box1, box2: [x1, y1, x2, y2] 格式的矩形框，(x1,y1)为左上角坐标，(x2,y2)为右下角坐标
    threshold: 重叠比例阈值，默认0.5
    
    返回:
    bool: 如果重叠比例超过阈值则返回True，否则返回False
    """
    overlap_area, ratio1, ratio2 = compute_overlap(box1, box2)

    return overlap_area > 0 and (ratio1 > threshold or ratio2 > threshold)

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