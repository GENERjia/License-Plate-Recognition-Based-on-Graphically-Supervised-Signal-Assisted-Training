import random
from pathlib import Path

import cv2
# from imutils import paths
import numpy as np
from PIL import ImageFont, Image, ImageDraw
from torch.utils.data import *

# from PIL import Image as Im


provincelist = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"]

wordlist = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"]

CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',

    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',

    '云', '京', '冀', '吉', '宁', '川', '新', '晋', '桂', '沪', '津', '浙',
    '渝', '湘', '琼', '甘', '皖', '粤', '苏', '蒙', '藏', '西',
    '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑',

    '-'
]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
# print(CHARS_DICT)


class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None, type='point'):
        random.seed(10)
        self.img_dir = img_dir
        self.img_paths = []
        # for i in range(len(img_dir)):
        #     self.img_paths += [el for el in paths.list_images(img_dir[i])]
        data_root = Path(img_dir).parent.parent

        lines = Path(img_dir).open('r').readlines()
        for line in lines:
            info = line.rstrip('\n').split('/')
            dir_, image_name = info[0], info[1]

            name = image_name.rstrip('.jpg').split('-')
            _, _, no = name[2], name[3], name[4]

            no = list(map(int, no.split('_')))

            text = provincelist[no[0]]
            for n in no[1:]:
                text += wordlist[n]

            self.img_paths.append((data_root / dir_/ type / image_name, text))

        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename, imgname = self.img_paths[index]
        # print(filename, imgname)
        Image = cv2.imread(str(filename))
        # Image = cv2.imdecode(np.fromfile(str(filename), dtype=np.uint8), cv2.IMREAD_COLOR)
        # Image = Im.open(str(filename))
        # cv2.imshow('image',Image)
        # cv2.waitKey()

        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)
        # basename = os.path.basename(filename)
        # imgname, suffix = os.path.splitext(basename)
        # imgname = imgname.split("-")[0].split("_")[0]
        label = list()
        for c in imgname:
            # one_hot_base = np.zeros(len(CHARS))
            # one_hot_base[CHARS_DICT[c]] = 1
            label.append(CHARS_DICT[c])

        if len(label) == 8:
            if self.check(label) == False:
                print(imgname)
                assert 0, "Error label ^~^!!!"

        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        else:
            return True
        
        
class LPRDataLoader_with_image_label(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None, type='point'):
        random.seed(10)
        self.img_dir = img_dir
        self.img_paths = []
        # for i in range(len(img_dir)):
        #     self.img_paths += [el for el in paths.list_images(img_dir[i])]
        data_root = Path(img_dir).parent.parent

        lines = Path(img_dir).open('r').readlines()
        for line in lines:
            info = line.rstrip('\n').split('/')
            dir_, image_name = info[0], info[1]

            name = image_name.rstrip('.jpg').split('-')
            _, _, no = name[2], name[3], name[4]

            no = list(map(int, no.split('_')))

            text = provincelist[no[0]]
            for n in no[1:]:
                text += wordlist[n]

            self.img_paths.append((data_root / dir_ / type / image_name, text))

        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename, imgname = self.img_paths[index]
        image = cv2.imread(str(filename))

        height, width, _ = image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            image = cv2.resize(image, self.img_size)
        image = self.PreprocFun(image)
        label = list()
        for c in imgname:
            label.append(CHARS_DICT[c])

        if len(label) == 8:
            if self.check(label) == False:
                print(imgname)
                assert 0, "Error label ^~^!!!"

        # label_im = self.label_image(imgname).astype('float32') / 255.
        label_im = self.label_image(imgname)
        # label_im = np.transpose(label_im, (2, 0, 1))
        # label_im = self.transform(label_im)

        return image, label, len(label), label_im

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        else:
            return True
    
    def label_image(self, label):

        fontPath = r"C:\Windows\Fonts\simhei.ttf"
        img = Image.new("L", (94 * 2, 24 * 2), 0)

        for i, c in enumerate(label):
            if i == 0:
                font = ImageFont.truetype(fontPath, 25)
            else:
                font = ImageFont.truetype(fontPath, 40)

            dr = ImageDraw.Draw(img)
            x, y = (7 + i * 23), 4
            if i == 0:
                x = 1
                y = 10
            if i >= 2:
                x = (19 + i * 23)
            dr.text((x, y), c, font=font, fill=255)

        img = img.resize(self.img_size, Image.BICUBIC)
        # img.show()
        img = np.array(img, dtype='uint8')[None].astype('float32') / 255.
        return img
