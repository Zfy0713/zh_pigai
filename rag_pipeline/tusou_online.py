import time;
import uuid
import hashlib
import json
import csv
import base64
from base64 import b64encode
import requests
from bs4 import BeautifulSoup
from urllib3 import encode_multipart_formdata
from PIL import Image
from io import BytesIO
import io
import os
import sys
import cv2
import numpy as np
import ast


app_key = "zixueyunpa8bdbe6" #"47ea14770d"
app_secret = "794857dd85e30821afb5c7e96fb5884d"#"850ef85574a93f9d57cbefd7ea700147"

def load_csv_2_dict(csv_path):
    # 增加字段大小限制
    csv.field_size_limit(sys.maxsize)
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

def valid_image_size(url):
    response = requests.get(url)
    if response.status_code == 200:
        image_file = io.BytesIO(response.content)
        # 打开图片并获取尺寸
        with Image.open(image_file) as img:
            width, height = img.size
        if width < 20 or height < 20:
            return False
        return True
    else:
        raise Exception(f"Failed to download image from {url}")   

def compress_image_in_memory(input_image_bytes, postfix, quality=85):
    """
    在内存中压缩图片。
    
    :param input_image_bytes: 输入图片的字节数据
    :param quality: 图片质量（1-100），值越低表示压缩率越高，默认值为85
    :return: 压缩后图片的字节数据
    """
    # 将输入的字节数据转换为BytesIO对象
    input_image_stream = io.BytesIO(input_image_bytes)
    
    # 打开原始图片
    with Image.open(input_image_stream) as img:
        # 创建一个BytesIO对象用于保存压缩后的图片
        output_image_stream = io.BytesIO()
        # 保存图片时设置quality参数控制压缩比
        if postfix.lower() == ".png":
            if img.mode == "RGBA":
                img = img.convert("RGB")
                img.save(output_image_stream, format='JPEG', quality=40)
        elif postfix.lower() in [".jpg", ".jpeg"]:
            img.save(output_image_stream, format='JPEG', quality=quality)
        # 获取压缩后的图片字节数据
        compressed_image_bytes = output_image_stream.getvalue()
        
    return compressed_image_bytes

def get_image_file_size_by_reading(file_path):
    """
    通过读取文件内容获取图片文件的大小。
    
    :param file_path: 图片文件路径
    :return: 文件大小（字节）
    """
    with open(file_path, 'rb') as f:
        content = f.read()
        return len(content)

def get_image_file_size_in_memory(image_bytes):
    """
    获取内存中图片的大小。
    
    :param image_bytes: 图片的字节数据
    :return: 文件大小（字节）
    """
    image_stream = io.BytesIO(image_bytes)
    return len(image_bytes)

def get_image_extension(url):
    # 获取文件名部分
    filename = os.path.basename(url)
    # 分割文件名和扩展名
    _, extension = os.path.splitext(filename)
    return extension

def image_url_to_base64(url):
    # 获取图片内容
    response = requests.get(url)
    if response.status_code == 200:
        image_size = get_image_file_size_in_memory(response.content)
        # print("image_size:", image_size/1024/1024)
        if image_size > 1024 * 1024:
            postfix = get_image_extension(url)
            print("comprise before.................image_size:", image_size/1024/1024)
            compressed_image_bytes = compress_image_in_memory(response.content, postfix, quality=70)
            print("comprise after...................image_size:", get_image_file_size_in_memory(compressed_image_bytes) / (1024 * 1024))
            base64_encoded_image = base64.b64encode(compressed_image_bytes).decode('utf-8')
        else:
            base64_encoded_image = base64.b64encode(response.content).decode('utf-8')
        return str(base64_encoded_image)
    else:
        raise Exception(f"Failed to download image from {url}")

def xiaohou_http_send(req_id, image, words, image_url):
    try:
        time_stamp = str(int(time.time()))
        nonce = str(uuid.uuid4())
        sign_str = 'X-TAL-Timestamp=' + time_stamp + \
                    '&X-TAL-Nonce=' + nonce + \
                    '&X-TAL-AppKey=' + app_key + \
                    '&app_secret=' + app_secret
        sign_result = hashlib.md5(sign_str.encode(encoding='utf-8')).hexdigest()
        uid = str(uuid.uuid4())
        # print(uid)
        # question
        question_header = {
            'X-TAL-Timestamp': time_stamp,
            'X-TAL-Nonce': nonce,
            'X-TAL-AppKey': app_key,
            'X-TAL-Sign': sign_result,
            'X-Request-ID':uid
        }

        use_image = False
        use_words = False
        question_body = {}
        if image is not None:
            use_image = True
            # image_file = ('{}.jpg'.format(str(uuid.uuid1())), image)
            question_body = {'image_base64': image, 'content': "", 'top_num': 10}
        if image is None and words is not None:
            use_words = True
            question_body = {'words': words,"size":1}
        if image_url is not None:
            use_image = True
            c = image_url_to_base64(image_url)
            question_body = {
                'image_base64': str(c),
                'content':"",
                'top_num':10
            }

        question_header['Content-Type'] = "application/json" #encode_data[1]

        search_url = "https://ask.mathcn.com/pdopen/api/tpp/search/content"
        if use_image is True:
            question_ret = requests.post(search_url, data=json.dumps(question_body), headers=question_header)


        return json.loads(question_ret.content.decode("utf-8"))
    except Exception as e:
        return {'error_code': -1, 'trace_id': str(uuid.uuid1()), 'msg': 'http send except msg:{}'.format(e)}

if __name__ == '__main__':

    import argparse
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--jiaozheng_dir", type=str)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--ocr_column", type=str, default="ocr", help="ocr column name")

    args = parser.parse_args()
    input_path = args.input_path
    save_path = args.save_path
    jiaozheng_dir = args.jiaozheng_dir
    top_k = args.top_k
    ocr_column = args.ocr_column

    data = load_csv_2_dict(input_path)

    targets = []
    title_set = set()
    logger.info(f"tusou begin!")
    for idx, d in enumerate(data, start=1):
        logger.info(f"processing data {idx}/{len(data)}...")
        
        
        url = d['img_url'] if 'img_url' in d else d['url']
        title = d['title'] if 'title' in d else ''

        try:
            query = d[ocr_column]
        except Exception as e:
            logger.error(f"Error in getting ocr column: {e}")
            target.append(json.dumps({"url": url}))

            continue

        if title not in title_set:
            title_set.add(title)
        else:
            query = query.replace(title, '')
        t = {}
        
        # image_url = d["img_url"]

        t["url"] = url
        t["input_content"] = query
        t["source"] = "tusou"

        image_name = os.path.split(url)[-1]
        image_path = os.path.join(jiaozheng_dir, image_name)
        vertices = ast.literal_eval(d["vertices"])
        image = Image.open(image_path)
        min_x = min(v['x'] for v in vertices)
        min_y = min(v['y'] for v in vertices)
        max_x = max(v['x'] for v in vertices)
        max_y = max(v['y'] for v in vertices)
        cropped_image = image.crop((min_x, min_y, max_x, max_y))
        # cropped_array = np.array(cropped_image)
        # _, buffer = cv2.imencode('.jpg', cropped_array)
        # binary_data = buffer.tobytes()
        img_byte_arr = io.BytesIO()
        cropped_image.save(img_byte_arr, format='PNG')
        binary_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        # t["source"] = "tusou"

        time.sleep(1)
        try:
            res = xiaohou_http_send("123", binary_data, None, None)
            # print(res)
            # question_ocr = res["data"]["image_txt"]
            questions = res["data"]["question_arr"]
        except Exception as e:
            print(e)
            print("search - here")
            try:
                time.sleep(1)
                res = xiaohou_http_send("123", binary_data, None, None)
                questions = res["data"]["question_arr"]
            except:
                print(e)
                questions = []
        print("length:", len(questions))
        # for jieti rag
        if len(questions) >=5:
            questions = questions[:5]
        arr = []
        for q in questions:
            print(q)
            qes = {}
            qid = q["question_id"]
            subject_id = q["subject_id"]
            if subject_id != 1: # 语文
                continue
            htmlq = q["question"]
            print("bqxxxxxxxxxxxxxxxxxx")
            print(htmlq)
            print("eqxxxxxxxxxxxxxxxxxx")
            htmla = q["answer"]
            html_analysis = ""
            if q["hint"] !="":
                html_analysis = q["hint"] 
            else:
                html_analysis = ""
            soup_content = BeautifulSoup(htmlq, 'html.parser')
            soup_answer = BeautifulSoup(htmla, 'html.parser')
            soup_analysis = BeautifulSoup(html_analysis, 'html.parser')
            content = ""
            if "<img" in htmlq:
                q_images = soup_content.find_all("img")
                p_ocr_list = soup_content.find_all(attrs={'class' : 'ocr_text_invisible'})
                if len(p_ocr_list) != 0:
                    content += soup_content.get_text()
                    content = content.replace("如图所示", "")
                else:
                    content += soup_content.get_text()
                    if len(q_images) > 0:
                        q_image_urls = [q.get("src") for q in q_images]
                        try:
                            for url in q_image_urls:
                                time.sleep(1)
                                if not valid_image_size(url) or url.endswith(".gif"):
                                    continue
                                res = xiaohou_http_send("123", None, None, url)
                                print(url)
                                print("qes-----------------------------------")
                                if res["data"]["image_txt"] !="":
                                    content += res["data"]["image_txt"]
                        except Exception as e:
                            print(e)
                            print("here qes")
                            print(res)
                            try:
                                for url in q_image_urls:
                                    time.sleep(1)
                                    if not valid_image_size(url) or url.endswith(".gif"):
                                        continue                                    
                                    res = xiaohou_http_send("123", None, None, url)
                                    if res["data"]["image_txt"] !="":
                                        content += res["data"]["image_txt"]
                            except:
                                content += ""
            else:
                content = soup_content.get_text().strip()

            answer = ""
            print("baxxxxxxxxxxxxxxxxxx")
            print(htmla)
            print("eaxxxxxxxxxxxxxxxxxx")
            if "<img" in htmla:
                a_images = soup_answer.find_all("img")
                if len(a_images) > 0:
                    a_image_urls = [a.get("src") for a in a_images]
                    try:
                        for url in a_image_urls:
                            time.sleep(1)
                            if not valid_image_size(url) or url.endswith(".gif"):
                                continue
                            print(url)
                            print("ans-----------------------------------")                                                         
                            res = xiaohou_http_send("123", None, None, url)
                            if res["data"]["image_txt"] !="":
                                answer += res["data"]["image_txt"]
                    except Exception as e:
                        print(e)
                        print("here ans")
                        print(res)
                        try:
                            for url in a_image_urls:
                                time.sleep(1)
                                if not valid_image_size(url) or url.endswith(".gif"):
                                    continue                                
                                res = xiaohou_http_send("123", None, None, url)
                                if res["data"]["image_txt"] !="":
                                    answer += res["data"]["image_txt"]
                        except:
                            answer = ""
            else:
                answer = soup_answer.get_text().strip()
            answer = answer.replace("题拍拍", "").replace("O题拍拍","")
            answer = answer.replace("一每天进步一点点一", "").replace("-每天进步一点点-", "").replace("一 每天进步一点点一", "")
            # if answer == "" : continue
            analysis = ""
            if "<img" in html_analysis:
                ana_images = soup_analysis.find_all("img")
                if len(ana_images) > 0:
                    ana_image_urls = [ana.get("src") for ana in ana_images]
                    try:
                        for url in ana_image_urls:
                            time.sleep(1)
                            if not valid_image_size(url) or url.endswith(".gif"):
                                continue                             
                            print(url)
                            print("analysis-----------------------------------")                            
                            res = xiaohou_http_send("123", None, None, url)
                            if res["data"]["image_txt"] !="":
                                analysis += res["data"]["image_txt"]
                    except Exception as e:
                        print(e)
                        print("here ansis")
                        print(res)
                        try:
                            for url in ana_image_urls:
                                time.sleep(1)
                                res = xiaohou_http_send("123", None, None, url)
                                if res["data"]["image_txt"] !="":
                                    analysis += res["data"]["image_txt"]
                        except:
                            analysis = ""
            else:
                analysis = soup_analysis.get_text().strip()
                if analysis.strip() == "【解析】":
                    analysis = ""
                if analysis == "":
                    if answer !="" and "解析:" in answer:
                        idx = answer.find("解析:")
                        analysis = answer[idx:]
            qes["content"] = content.strip()
            qes["answer"] = answer.strip()
            qes["analysis"] = analysis.strip()
            qes["id"] = qid
            qes["score"] = q["cal_match_percent"]
            # print(qes)
            arr.append(qes)
        t["rag"] = arr
        targets.append(json.dumps(t, ensure_ascii=False))

    
    with open(save_path, "w", encoding="utf-8") as f:
        for d in targets:
            f.write(d + '\n')