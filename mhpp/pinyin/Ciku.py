import re, os, sys
import json

class Ciku:
    def __init__(self, ciku_path):
        self.ciku_path = ciku_path
        if ciku_path.endswith('.json'):
            with open(ciku_path, 'r', encoding='utf-8') as f:
                self.pinyin_dict = json.load(f)
        else:
            self.pinyin_dict = self.load_ciku()

    @staticmethod
    def _remove_tones(pinyin):
    # 带声调到不带声调的字符映射
        tone_map = {
            'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
            'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
            'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
            'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
            'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
            'ǖ': 'v', 'ǘ': 'v', 'ǚ': 'v', 'ǜ': 'v', 'ü': 'v'
        }
        return ''.join(tone_map.get(char, char) for char in pinyin)

    def load_ciku(self):
        if not os.path.exists(self.ciku_path):
            raise FileNotFoundError(f"Ciku file not found: {self.ciku_path}")
        
        pinyin_dict = {}
        with open(self.ciku_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(':')
                if len(parts) != 2:
                    continue
                hanzi, pinyin = parts[0].strip(), parts[1].strip()
                if pinyin not in pinyin_dict:
                    pinyin_dict[pinyin] = []
                pinyin_dict[pinyin].append(hanzi)
        return pinyin_dict

    def drop_duplicate(self):
        # 去除重复的拼音和汉字对
        unique_dict = {}
        for pinyin, hanzi_list in self.pinyin_dict.items():
            unique_hanzi = list(set(hanzi_list))
            unique_dict[pinyin] = unique_hanzi
        self.pinyin_dict = unique_dict
    
    def save_to_json(self, save_path):
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.pinyin_dict, f, ensure_ascii=False, indent=4)
        print(f"拼音字典已保存到 {save_path}")

    def merge_ciku(self, other_ciku):
        if not isinstance(other_ciku, Ciku):
            raise TypeError("other_ciku must be an instance of Ciku")
        
        for pinyin, hanzi_list in other_ciku.pinyin_dict.items():
            if pinyin not in self.pinyin_dict:
                self.pinyin_dict[pinyin] = hanzi_list
            else:
                self.pinyin_dict[pinyin].extend(hanzi_list)
                self.pinyin_dict[pinyin] = list(set(self.pinyin_dict[pinyin]))

    def delete_hanzi(self, hanzi):
        for pinyin, hanzi_list in self.pinyin_dict.items():
            if hanzi in hanzi_list:
                hanzi_list.remove(hanzi)
                if not hanzi_list:
                    del self.pinyin_dict[pinyin]

    def get_hanzi(self, pinyin):
        return self.pinyin_dict.get(pinyin, [])



if __name__ == "__main__":
    ciku_path = '/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v2/pinyin_large/large_pinyin_v2.txt'
    ciku1 = Ciku(ciku_path)
    ciku1.drop_duplicate()
    save_path = '/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v2/pinyin_large/pinyin_dict.json'
    ciku1.save_to_json(save_path)

    # 示例：合并两个词库
    ciku2_path = '/mnt/pfs_l2/jieti_team/APP/zhangfengyu/zhangfengyu/Correct_model/pigai_pipeline/pigai_v2/pinyin_large/pinyin_supplement.txt'
    ciku2 = Ciku(ciku2_path)
    ciku2.drop_duplicate()
    ciku1.merge_ciku(ciku2)
    ciku1.save_to_json(save_path)

    # 示例：删除特定汉字
    ciku1.delete_hanzi('拥带')
    ciku1.save_to_json(save_path)
