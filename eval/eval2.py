import argparse
import glob
import json
import numpy as np
import os
import cv2
import re
import csv
import sys
# import torch
import shutil
import random
from pathlib import Path
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from collections import defaultdict
# from pycocotools.coco import COCO
from ..utils.utils import load_json, load_csv, csv2xlsx
from PIL import Image, ImageDraw, ImageOps

yy = {}

def load_gt(path, pic_path, rand_path=None):
    gt_json = load_json(path)
    gt_kong = {}
    gt_question = {}
    gt_lxdk = {}
    cnt=0
    if rand_path is None:
        _rand = None
    elif rand_path[-3:] == 'csv':
        __rand = load_csv(rand_path)
        _rand = []
        for i in __rand:
            for j in i:
                if j and len(j)>10:
                    _rand.append(j.split('/')[-1])
    elif rand_path[-3:] == 'txt':
        with open(rand_path, 'r', encoding='utf-8') as f:
            _rand = [item.strip().split('/')[-1] for item in f.readlines()]
    for i in gt_json['taskDataList']:
        image = i['dataName']
        if _rand and  (image not in _rand):
            continue
        if i['jsonMarkData'] == None:
            # gt[image] = _gt
            cnt+=1
            continue
        
        # if not os.path.exists(f"{pic_path}{image}"):
        #     print(image, "图片不存在")
        #     continue
            
        img_dir = f"{pic_path}"
        
        for f in os.listdir(img_dir):
            if f.lower().endswith(image.lower()):
                image = f
                break
        if not os.path.exists(f"{pic_path}{image}"):
            print(f"{pic_path}{image} 图片不存在")
            continue
        pic = Image.open(f"{pic_path}{image}")
        width, height = pic.size
        _gt_kong = []
        _gt_question = []
        _gt_lxdk = []
        
        # cnt = 0
        for j in i['jsonMarkData']:
            __gt = {}
            points = []
            for k in j['markGraph']['points']:
                points.append((width/2+k['x'], height/2-k['y']))
            __gt['polygon'] = Polygon(points)
            __gt['label'] = j['markLabel']['value']['text']
            __gt['flag1'] = 0
            __gt['flag2'] = 0
            __gt['flag3'] = 0
            # __gt['id'] = cnt
            # cnt += 1
            if __gt['label'] == "整题":
                if len(j['markAttrList'])==0:
                    continue
                for k in j['markAttrList']:
                    if k['name'] == '题目类别':
                        __gt['type'] = k['value']
                        __gt['type2'] = k['value']
                    elif k['name'] == '题型':
                        __gt['type'] += '-' + k['value']
                    elif k['name'] == '知识点':
                        __gt['type2'] += '-' + k['value']
                    # if __gt['type2'] == '可批改题型-':
                        
                        # print(image)
                        # input()

                __gt['kong_total'] = 0
                __gt['kong_recall'] = 0
                __gt['kong_acc'] = 0
                __gt['kong_wuzhao'] = 0
                # if __gt['type'].startswith("不支持") or __gt['type'].startswith("无效"):
                #     continue
            if __gt['label'] == "正确" or __gt['label'] == "错误":
                _gt_kong.append(__gt)
            elif __gt['label'] == "整题":
                _gt_question.append(__gt)
            elif __gt['label'] == "连续多空":
                _gt_lxdk.append(__gt)
            elif __gt['label'] != "未作答":
                print(__gt['label'],'!'*100)
                input()
        # if '_' in image:
        #     yy[image.split('_')[-1]] = image
        #     image = image.split('_')[-1]
        # else:
        yy[image]=image
        gt_kong[image] = _gt_kong
        gt_question[image] = _gt_question
        gt_lxdk[image] = _gt_lxdk
    return gt_kong, gt_question, gt_lxdk

def load_jiuzhang(path, pic_path, rand_path=None):
    _json = load_csv(path)
    jiuzhang = {}
    url = {}
    if rand_path is None:
        _rand = None
    elif rand_path[-3:] == 'csv':
        __rand = load_csv(rand_path)
        _rand = []
        for i in __rand:
            for j in i:
                if j and len(j)>10:
                    _rand.append(j.split('/')[-1])
    elif rand_path[-3:] == 'txt':
        with open(rand_path, 'r', encoding='utf-8') as f:
            _rand = [item.strip().split('/')[-1] for item in f.readlines()]
    for i in _json:
        image = i[0].split('/')[-1]
        if image not in yy:
            continue
        if not Path(f"{pic_path}{yy[image]}").exists():
            continue
        if _rand and  (image not in _rand):
            continue
        i[1] = re.sub(r"'", '"', str(i[1]))
        print(image)
        print(i[1])
        i[1] = json.loads(str(i[1]))
        for j in i[1]:
            j['point'] = Point(j['x'], j['y'])
            j.pop('x')
            j.pop('y')
        jiuzhang[image] = i[1]
        url[image] = i[0]
    return jiuzhang, url

def cal(q_type, kong_total, kong_recall, kong_acc, kong_wuzhao, question_total, question_recall, question_acc, question_wuzhao):
    kong_jingque = kong_acc/kong_recall if kong_recall > 0 else 0
    if kong_total == 0:
        return [q_type,0,0,0,0,0,0,0,0]
    question_jingque = question_acc/question_recall if question_recall > 0 else 0
    
    return [q_type, kong_jingque, kong_recall/kong_total, kong_wuzhao/(kong_wuzhao+kong_total), kong_acc/kong_total, 
            question_jingque, question_recall/question_total, question_wuzhao/(question_wuzhao+question_total), question_acc/question_total]

def judge(area_a, area_b):
    if not area_a.is_valid or not area_a.is_valid:
        return False
    if area_a.contains(area_b) or area_b.contains(area_a):
        return True
    try:
        if min(area_a.area, area_b.area) == 0:
            return False
        inter = area_a.intersection(area_b)
        if inter.is_empty:
            return False
        return inter.area/min(area_a.area, area_b.area) > 0.1
    except Exception:
        print("fuc judge error")
        return False
    
def process(jiuzhang, url, kong_gt, question_gt, lxdk_gt, pic_path, out_path, mhpp_path = None):
    # shutil.rmtree(out_path)
    kong_total = 0
    kong_recall = 0
    kong_acc = 0
    kong_wuzhao = 0
    question_total = 0
    question_recall = 0
    question_acc = 0
    question_wuzhao = 0
    
    question_type_result = {}
    question_type2_result = {}
    
    # cnt = 0
    # total = 100
    url_list = []
    per_pic = [['图片','总空数','批改空数','批对空数','多批空数','总题数','批改题数','批对题数','多批题数']]
    select_url=[]
    # select=["可批改题型-句子", "可批改题型-字音", "可批改题型-文言文阅读", "可批改题型-现代文阅读", "可批改题型-理解性背诵默写", "可批改题型-直接性背诵默写", "可批改题型-诗歌鉴赏"]
    # select=["可批改题型-句子", "可批改题型-阅读", "可批改题型-填空", "可批改题型-圈选勾画题"]
    # select=["可批改题型-直接性背诵默写", "可批改题型-现代文阅读", "可批改题型-理解性背诵默写", "可批改题型-句子", "可批改题型-常识", "可批改题型-拼音","可批改题型-字音", "可批改题型-诗歌鉴赏", "可批改题型-实用性文本阅读", "可批改题型-文言文阅读"]
    # select=["可批改题型-直接性背诵默写", "可批改题型-理解性背诵默写", "可批改题型-现代文阅读", "可批改题型-常识"] # 1021
    # select=["可批改题型-直接性背诵默写", "可批改题型-理解性背诵默写", "可批改题型-现代文阅读", "可批改题型-常识", "可批改题型-句子",'可批改题型-拼音', '可批改题型-字音','可批改题型-实用性文本阅读'] # 1022

    select = ["可批改题型-直接性背诵默写", "可批改题型-理解性背诵默写"]

    zhengye_correct = 0
    total_page = 0

    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True, exist_ok=True)
    for i in kong_gt:
        if i not in jiuzhang:
            print(i,"not in jiuzhang")
            continue
        if not i in kong_gt or not i in question_gt or len(kong_gt[i]) == 0 or len(question_gt[i]) == 0:
            print(i,"not in gt")
            continue
        # if cnt >= total or random.random() < 0.5:
        #     continue
        # cnt += 1
        
        url_list.append(url[i])
        
        _kong_total = 0
        _kong_recall = 0
        _kong_acc = 0
        _kong_wuzhao = 0
        _question_total = 0
        _question_recall = 0
        _question_acc = 0
        _question_wuzhao = 0
        _kong_loupan = []
        _kong_cuopan = []
        
        
        
        for j in jiuzhang[i]:
            flag = False
            for k in kong_gt[i]:
                if k['polygon'].contains(j['point']):# or j['point'].distance(k['polygon']) < 15:
                    flag = True
                    if j['result'] == 1:
                        k['flag1'] += 1 #批对
                    if j['result'] == 0:
                        k['flag2'] += 1 #批错
                    break
            if not flag:
                _kong_wuzhao += 1
                for k in question_gt[i]:
                    if k['polygon'].contains(j['point']):# or j['point'].distance(k['polygon']) < 10:
                        k['flag1'] = 1 #题误召
                        
        for j in jiuzhang[i]:
            for k in lxdk_gt[i]:
                if k['polygon'].contains(j['point']):
                    for l in kong_gt[i]:
                        if l['flag1'] + l['flag1'] ==0 and judge(k['polygon'], l['polygon']):
                            if j['result'] == 1:
                                l['flag1'] += 1 #批对
                            if j['result'] == 0:
                                l['flag2'] += 1 #批错
                    break
        
        for j in kong_gt[i]:
            _kong_total += 1
            
            if j['flag1'] + j['flag2'] == 0:
                _kong_loupan.append(j['polygon'])  #空漏判
            else:
                _kong_recall += 1  #空召回
                if j['flag1'] * j['flag2'] == 0 and ((j['flag1']>0 and j['label']=='正确')or(j['flag2']>0 and j['label']=='错误')):  #空判对
                    _kong_acc += 1
                else:  #空判错
                    _kong_wuzhao += max(j['flag1'] + j['flag2'] - 1, 0)
                    _kong_cuopan.append(j['polygon'])
            flag=True
            for k in question_gt[i]:
                if judge(k['polygon'], j['polygon']):
                    k['kong_total'] += 1
                    if j['flag1'] + j['flag2'] == 0:
                        k['flag2'] = 1 #题漏召
                        k['flag3'] = 1 #题批错
                    else:
                        k['kong_recall'] += 1
                        if j['flag1'] * j['flag2'] != 0 or (j['flag2']>0 and j['label']=='正确') or (j['flag1']>0 and j['label']=='错误'):
                            k['flag3'] = 1 #题批错
                            if j['flag1'] + j['flag2'] > 1:
                                k['flag1'] = 1 #题误召
                                k['kong_wuzhao'] += j['flag1'] + j['flag2'] - 1
                        else:
                            k['kong_acc'] += 1
                    flag=False
                    break
            if flag:
                print(yy[i], "空不属于任何题")
                # input()
        
        # flag = True
        for j in question_gt[i]:
            # if (j['type'].startswith('不支持') or j['type'].startswith('无效题')) and j['kong_total']>0:
            #     print(i, j['kong_total'])
            #     input()
            # if j['type'].startswith("不支持") or j['type'].startswith("无效"):
            #     continue
            _question_total += 1
            if j['type'] not in question_type_result:
                question_type_result[j['type']] = {'kong_total': 0,'kong_recall': 0,'kong_acc': 0,'kong_wuzhao': 0,'question_total': 0,'question_recall': 0,'question_acc': 0,'question_wuzhao': 0}
            question_type_result[j['type']]['kong_total'] += j['kong_total']
            question_type_result[j['type']]['kong_recall'] += j['kong_recall']
            question_type_result[j['type']]['kong_acc'] += j['kong_acc']
            question_type_result[j['type']]['kong_wuzhao'] += j['kong_wuzhao']
            question_type_result[j['type']]['question_total'] += 1
            if j['flag1']:
                question_type_result[j['type']]['question_wuzhao'] += 1
                _question_wuzhao += 1
            if not j['flag2']:
                question_type_result[j['type']]['question_recall'] += 1
                _question_recall += 1
            if not j['flag3']:
                question_type_result[j['type']]['question_acc'] += 1
                _question_acc += 1
                
            # 知识点
            if j['type2'] not in question_type2_result:
                question_type2_result[j['type2']] = {'kong_total': 0,'kong_recall': 0,'kong_acc': 0,'kong_wuzhao': 0,'question_total': 0,'question_recall': 0,'question_acc': 0,'question_wuzhao': 0}
            question_type2_result[j['type2']]['kong_total'] += j['kong_total']
            question_type2_result[j['type2']]['kong_recall'] += j['kong_recall']
            question_type2_result[j['type2']]['kong_acc'] += j['kong_acc']
            question_type2_result[j['type2']]['kong_wuzhao'] += j['kong_wuzhao']
            question_type2_result[j['type2']]['question_total'] += 1
            if j['flag1']:
                question_type2_result[j['type2']]['question_wuzhao'] += 1
            if not j['flag2']:
                question_type2_result[j['type2']]['question_recall'] += 1
            if not j['flag3']:
                question_type2_result[j['type2']]['question_acc'] += 1
        if len(_kong_loupan) >= 0 or len(_kong_cuopan) > 0:
            image_np = cv2.imread(f"{pic_path}{yy[i]}").copy()
            # image_np = cv2.imread(f"{pic_path}{i}").copy()
            for j in _kong_loupan:
                exterior_coords = np.array(j.exterior.coords, np.int32)
                pts = exterior_coords.reshape((-1, 1, 2))
                cv2.polylines(image_np, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            for j in _kong_cuopan:
                exterior_coords = np.array(j.exterior.coords, np.int32)
                pts = exterior_coords.reshape((-1, 1, 2))
                # cv2.polylines(image_np, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
                cv2.polylines(image_np, [pts], isClosed=True, color=(255, 0, 0), thickness=3) # (255,0,0) for blue

                
            save_path = f"{out_path}{i}"
            
            cv2.imwrite(save_path, image_np)
            
        if len(_kong_loupan) + len(_kong_cuopan) > 2:    
            for j in question_gt[i]:
                if j['type2'] in select:
                    select_url.append([i])
                    break
        if _kong_total == _kong_acc:
            zhengye_correct += 1
        total_page += 1
        per_pic.append([i,_kong_total,_kong_recall,_kong_acc,_kong_wuzhao,_question_total,_question_recall,_question_acc,_question_wuzhao, _kong_total == _kong_acc])
        kong_total += _kong_total
        kong_recall += _kong_recall
        kong_acc += _kong_acc
        kong_wuzhao += _kong_wuzhao
        question_total += _question_total
        question_recall += _question_recall
        question_acc += _question_acc
        question_wuzhao += _question_wuzhao
        # print(_kong_total, _kong_recall, _kong_acc, kong_wuzhao)
        
    # for i in kong_gt:
    #     if i not in jiuzhang:
    #         kong_total += len(kong_gt[i])
    #         for j in kong_gt[i]:
    #             for k in question_gt[i]:
    #                 if judge(k['polygon'], j['polygon']):
    #                     k['kong_total'] += 1
    #                     break
            
    # for i in question_gt:
    #     if i not in jiuzhang:
    #         question_total += len(question_gt[i])
    #         for j in question_gt[i]:
    #             if j['type'] not in question_type_result:
    #                 question_type_result[j['type']] = {'kong_total': 0,'kong_recall': 0,'kong_acc': 0,'kong_wuzhao': 0,'question_total': 0,'question_recall': 0,'question_acc': 0,'question_wuzhao': 0}
    #             question_type_result[j['type']]['kong_total'] += j['kong_total']
    #             question_type_result[j['type']]['question_total'] += 1
           
    
    if mhpp_path:
        mhpp_list = load_csv(mhpp_path)
        select_result = [mhpp_list[0]]
        for i in mhpp_list:
            if i[1] in url_list:
                select_result.append(i)
        with open(f"{out_path}select_result.csv", 'w', encoding='utf-8', newline="") as f:
            w = csv.writer(f)
            w.writerows(select_result)
    with open(f"{out_path}select_url.csv", 'w', encoding='utf-8', newline="") as f:
        w = csv.writer(f)
        w.writerows(select_url)

    # csv2xlsx(f"{out_path}select_result.csv", f"{out_path}select_result.xlsx")

    with open(f"{out_path}per_pic.csv", 'w', encoding='utf-8', newline="") as f:
        w = csv.writer(f)
        w.writerows(per_pic)
        
    final_result = [['题型','空维度精确率','空维度召回率','空维度误召率','空维度正确率','题维度精确率','题维度召回率','题维度误召率','题维度正确率']]
    out_result = cal('总计', kong_total, kong_recall, kong_acc, kong_wuzhao, question_total, question_recall, question_acc, question_wuzhao)
    final_result.append(out_result)
    _kong_total = 0
    _kong_recall = 0
    _kong_acc = 0
    _kong_wuzhao = 0
    _question_total = 0
    _question_recall = 0
    _question_acc = 0
    _question_wuzhao = 0
    for i, j in sorted(question_type_result.items()):
        if i.startswith("可批改"):
            _kong_total += j['kong_total']
            _kong_recall += j['kong_recall']
            _kong_acc += j['kong_acc']
            _kong_wuzhao += j['kong_wuzhao']
            _question_total += j['question_total']
            _question_recall += j['question_recall']
            _question_acc += j['question_acc']
            _question_wuzhao += j['question_wuzhao']
    final_result.append(cal('可批改题型', _kong_total, _kong_recall, _kong_acc, _kong_wuzhao, _question_total, _question_recall, _question_acc, _question_wuzhao))
    
    
    for i, j  in sorted(question_type_result.items()):
        final_result.append(cal(i, j['kong_total'], j['kong_recall'], j['kong_acc'], j['kong_wuzhao'], j['question_total'], j['question_recall'], j['question_acc'], j['question_wuzhao']))
    final_result.append(['题型','总空数','召回空数','批对空数','误召空数','总题数','召回题数','批对题数','误召题数'])
    final_result.append(['总计', kong_total, kong_recall, kong_acc, kong_wuzhao, question_total, question_recall, question_acc, question_wuzhao])
    final_result.append(['可批改题型', _kong_total, _kong_recall, _kong_acc, _kong_wuzhao, _question_total, _question_recall, _question_acc, _question_wuzhao])
    for i, j  in sorted(question_type_result.items()):
        final_result.append([i, j['kong_total'], j['kong_recall'], j['kong_acc'], j['kong_wuzhao'], j['question_total'], j['question_recall'], j['question_acc'], j['question_wuzhao']])
    
    with open(f"{out_path}final_result.csv", 'w', encoding='utf-8', newline="") as f:
        w = csv.writer(f)
        w.writerows(final_result)
    print(out_result)
    
    #知识点
    final_result = [['题型','空维度精确率','空维度召回率','空维度误召率','空维度正确率','题维度精确率','题维度召回率','题维度误召率','题维度正确率']]
    out_result = cal('总计', kong_total, kong_recall, kong_acc, kong_wuzhao, question_total, question_recall, question_acc, question_wuzhao)
    final_result.append(out_result)
    _kong_total = 0
    _kong_recall = 0
    _kong_acc = 0
    _kong_wuzhao = 0
    _question_total = 0
    _question_recall = 0
    _question_acc = 0
    _question_wuzhao = 0
    for i, j  in sorted(question_type2_result.items()):
        if i.startswith("可批改"):
            _kong_total += j['kong_total']
            _kong_recall += j['kong_recall']
            _kong_acc += j['kong_acc']
            _kong_wuzhao += j['kong_wuzhao']
            _question_total += j['question_total']
            _question_recall += j['question_recall']
            _question_acc += j['question_acc']
            _question_wuzhao += j['question_wuzhao']
    final_result.append(cal('可批改题型', _kong_total, _kong_recall, _kong_acc, _kong_wuzhao, _question_total, _question_recall, _question_acc, _question_wuzhao))
    
    
    for i, j  in sorted(question_type2_result.items()):
        final_result.append(cal(i, j['kong_total'], j['kong_recall'], j['kong_acc'], j['kong_wuzhao'], j['question_total'], j['question_recall'], j['question_acc'], j['question_wuzhao']))
    final_result.append(['题型','总空数','召回空数','批对空数','误召空数','总题数','召回题数','批对题数','误召题数'])
    final_result.append(['总计', kong_total, kong_recall, kong_acc, kong_wuzhao, question_total, question_recall, question_acc, question_wuzhao])
    final_result.append(['可批改题型', _kong_total, _kong_recall, _kong_acc, _kong_wuzhao, _question_total, _question_recall, _question_acc, _question_wuzhao])
    for i, j  in sorted(question_type2_result.items()):
        final_result.append([i, j['kong_total'], j['kong_recall'], j['kong_acc'], j['kong_wuzhao'], j['question_total'], j['question_recall'], j['question_acc'], j['question_wuzhao']])
    

    final_result.append(['整页批对', '总页数', '整页准确率'])
    final_result.append([zhengye_correct, total_page, zhengye_correct/total_page])
    print(f"总页数: {total_page},整页批对/整页准确率: {zhengye_correct}({zhengye_correct/total_page*100:.2f}%)")

    with open(f"{out_path}final_result2.csv", 'w', encoding='utf-8', newline="") as f:
        w = csv.writer(f)
        w.writerows(final_result)
        


if __name__ == "__main__":
    pic_path = sys.argv[3]
    out_path = sys.argv[4]
    mhpp_path = sys.argv[5] if len(sys.argv) > 5 else None
    if len(sys.argv) > 6:
        rand_path = sys.argv[6]
    else:
        rand_path = None
    kong_gt, question_gt, lxdk_gt = load_gt(sys.argv[1], pic_path, rand_path)
    print(len(kong_gt))
    jiuzhang, url = load_jiuzhang(sys.argv[2], pic_path, rand_path)
    print("url len: ", len(url))
    process(jiuzhang, url, kong_gt, question_gt, lxdk_gt, pic_path, out_path, mhpp_path)
    