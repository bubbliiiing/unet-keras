import math
import os
from random import shuffle

import cv2
import keras
import numpy as np
from PIL import Image

from utils.utils import cvtColor, preprocess_input


class UnetDataset(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, batch_size, num_classes, train, dataset_path):
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        self.input_shape        = input_shape
        self.batch_size         = batch_size
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        images  = []
        targets = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.length
            name        = self.annotation_lines[i].split()[0]
            #-------------------------------#
            #   从文件中读取图像
            #-------------------------------#
            jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "Images"), name + ".png"))
            png         = Image.open(os.path.join(os.path.join(self.dataset_path, "Labels"), name + ".png"))
            #-------------------------------#
            #   数据增强
            #-------------------------------#
            jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)
            jpg         = preprocess_input(np.array(jpg, np.float64))
            png         = np.array(png)
            
            #-------------------------------------------------------#
            #   这里的标签处理方式和普通voc的处理方式不同
            #   将小于127.5的像素点设置为目标像素点。
            #-------------------------------------------------------#
            seg_labels  = np.zeros_like(png)
            seg_labels[png <= 127.5] = 1
            seg_labels  = np.eye(self.num_classes + 1)[seg_labels.reshape([-1])]
            seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

            images.append(jpg)
            targets.append(seg_labels)

        images  = np.array(images)
        targets = np.array(targets)
        return images, targets

    def on_epoch_end(self):
        shuffle(self.annotation_lines)
        
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label