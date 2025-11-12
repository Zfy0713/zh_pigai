import json, os, csv
from PIL import Image, ImageDraw, ImageFont
from ..utils.utils import load_jsonL, load_csv
import ast

FONT_PATH="/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v1/utils/NotoSansSC-Regular.ttf"

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

def draw_check_mark(draw_obj, x, y):
    length = 10
    draw_obj.line((x,y+20, x-length,y), fill='green', width=6)
    draw_obj.line((x,y+20, x+length+10,y-5), fill='green', width=6)

def draw_cross(draw_obj,x,y):
    x=x+10
    length = 10
    draw_obj.line((x-length,y-length,x+length,y+length),fill="red", width=6)
    draw_obj.line((x+length,y-length,x-length,y+length),fill="red", width=6)

def draw_rect(draw_obj, x,y,x1,y1, color, source_label):
    '''
    (x,y): 左上角坐标
    (x1,y1): 右下角坐标
    '''
    assert color in ['red', 'green']
    draw_obj.rectangle([x,y,x1,y1], outline=color, width=4)

    font_size = 20  # 设置字体大小
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        font = ImageFont.load_default()
    text_position = (x1 + 5, y - 15)  # 调整文字位置，保证在矩形框的右上角
    draw_obj.text(text_position, source_label, fill="blue", font=font)
    # if is_rag:
    #     text_position = (x1 + 5, y - 15)  # 调整文字位置，保证在矩形框的右上角
    #     draw_obj.text(text_position, "题库批改", fill="blue", font=font)
    # else:
    #     text_position = (x1 + 5, y - 15)  # 调整文字位置，保证在矩形框的右上角
    #     draw_obj.text(text_position, "大模型批改", fill="blue", font=font)

def draw(img_path, matched_box, save_img_path, rotate_angle=0):
    # 打开本地图片
    # img = Image.open(img_path)
    if os.path.exists(save_img_path):
        img = Image.open(save_img_path)
    elif os.path.exists(img_path):
        img = Image.open(img_path)
    else:
        raise FileNotFoundError(f"The file at path {img_path} does not exist.")    
    # 消除自动旋转

    # 创建绘图对象
    draw = ImageDraw.Draw(img)

    if not isinstance(matched_box, list):
        matched_box = ast.literal_eval(matched_box)
    # 遍历坐标信息并绘制对勾或叉
    print(f"matched_box len: {len(matched_box)}")
    for item in matched_box:
        result = int(item['result'])
        source_label = item['source']
        x = item['x']
        y = item['y']
        if result == 1:
            print("draw one mark!")
            # draw_check_mark(draw,x,y)
            draw_rect(draw, *item['box'], color='green', source_label=source_label)
        elif result == 0:
            print("draw one cross!")
            # draw_cross(draw,x,y)
            draw_rect(draw, *item['box'], color='red', source_label=source_label)
        elif result == 9:
            continue

    # 保存或显示处理后的图片
    # img.show() # 或者 img.save("output.jpg")
    if not(save_img_path.endswith('.jpg') or save_img_path.endswith('.jpeg')):
        save_img_path = save_img_path + '.jpg'
    img.save(save_img_path)

def load_matched_box(draw_path, source_label):
    '''
    draw_path: 批改结果 draw.csv路径
    source_label: 批改来源
    '''
    if not os.path.exists(draw_path):
        return []
    data = load_csv(draw_path)
    line = data[0]
    matched_box = line[1]

    if matched_box:
        matched_box = ast.literal_eval(matched_box)
        for box in matched_box:
            box.update({
                "source": source_label
            })
    else:
        matched_box = []
    return matched_box



def merge_box(box_list1: list, box_list2: list) -> list:
    '''
    优先级 box_list1 > box_list2
    '''
    ### 去掉未批改的框
    box_list1 = [box for box in box_list1 if box['result']!=9]
    box_list2 = [box for box in box_list2 if box['result']!=9]

    if not box_list1:
        print(f"merged lens: {len(box_list2)}")
        return box_list2
    for box2 in box_list2:
        if not any(judge_overlap(existing_box['box'], box2['box'], 0.5) for existing_box in box_list1):
            box_list1.append(box2)
    print(f"merged lens: {len(box_list1)}")
    return box_list1

def merge_n(*box_lists):
    """
    支持按优先级顺序合并任意数量的 box 列表
    参数：
        *box_lists: 多个批改结果列表，第一个参数优先级最高
    返回：
        合并后的批改框列表
    """
    if not box_lists:
        return []

    merged = box_lists[0]
    for box in box_lists[1:]:
        merged = merge_box(merged, box)  # 保留已合并结果的优先级
    return merged


def process_draw(args):
    pigai_dir = args.pigai_dir ### pigai_dir: 批改完整文件夹，包含jiaozheng, model_output, rag, smartmatch等等

    img_name = os.path.split(pigai_dir)[-1]
    img_path = os.path.join(pigai_dir, "jiaozheng", img_name)

    model_draw_path = os.path.join(pigai_dir, "model_output", "draw.csv")
    pinyin_draw_path = os.path.join(pigai_dir, "pinyin_ciku", "draw.csv")
    rag_draw_path = os.path.join(pigai_dir, "rag", "draw.csv")
    smartmatch_draw_path = os.path.join(pigai_dir, "smart_match", "draw.csv")

    # 优先级：教辅整页-->词库-->题库-->模型
    smartmatch_draw = load_matched_box(smartmatch_draw_path, source_label='整页')
    pinyin_draw = load_matched_box(pinyin_draw_path, source_label="拼音")
    rag_draw = load_matched_box(rag_draw_path, source_label='题库')
    model_draw = load_matched_box(model_draw_path, source_label='模型')

    print(len(smartmatch_draw), '\n', len(pinyin_draw), '\n', len(rag_draw), '\n', len(model_draw))

    merged_box = merge_n(smartmatch_draw, pinyin_draw, rag_draw, model_draw) ### 合并所有批改结果
    print(f"merged box len: {len(merged_box)}")
    save_merged_draw_path = args.save_merged_draw_path
    save_img_path = args.save_img_path
    if merged_box:
        draw(img_path=img_path, matched_box=merged_box, save_img_path=save_img_path)
        print(f"pigai results saved to {save_img_path}")

        with open(save_merged_draw_path, 'w', encoding='utf-8', newline="") as f:
            w=csv.writer(f)
            w.writerow([img_name, merged_box])
        print(f"merged draw saved to {save_merged_draw_path}")

    else:
        print("No pigai results found.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pigai_dir', type=str)
    parser.add_argument('--save_img_path', type=str)
    parser.add_argument('--save_merged_draw_path', type=str)
    
    args = parser.parse_args()
    process_draw(args)


