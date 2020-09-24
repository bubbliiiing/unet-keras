import keras
import tensorflow as tf
import numpy as np
from PIL import Image
from random import shuffle
from keras import backend as K
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from utils.metrics import f_score
import cv2         

def dice_loss_with_CE(beta=1, smooth = 1e-5):
    def _dice_loss_with_CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[...,:-1] * K.log(y_pred)
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))

        tp = K.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        fp = K.sum(y_pred         , axis=[0,1,2]) - tp
        fn = K.sum(y_true[...,:-1], axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score
        # dice_loss = tf.Print(dice_loss, [dice_loss, CE_loss])
        return CE_loss + dice_loss
    return _dice_loss_with_CE

def CE():
    def _CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[...,:-1] * K.log(y_pred)
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))
        # dice_loss = tf.Print(CE_loss, [CE_loss])
        return CE_loss
    return _CE

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def letterbox_image(image, label , size):
    image = image.convert("RGB")
    label = Image.fromarray(np.array(label))

    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    label = label.resize((nw,nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w-nw)//2, (h-nh)//2))
    return new_image, new_label


class Generator(object):
    def __init__(self,batch_size,train_lines,image_size,num_classes):
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.num_classes = num_classes

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        image = image.convert("RGB")
        label = Image.fromarray(np.array(label))

        h, w = input_shape
        # resize image
        rand_jit1 = rand(1-jitter,1+jitter)
        rand_jit2 = rand(1-jitter,1+jitter)
        new_ar = w/h * rand_jit1/rand_jit2

        scale = rand(0.5, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        label = label.convert("L")
        
        # flip image or not
        flip = rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        return image_data,label
        
    def generate(self, random_data = True):
        i = 0
        length = len(self.train_lines)
        inputs = []
        targets = []
        while True:
            if i == 0:
                shuffle(self.train_lines)
            annotation_line = self.train_lines[i]
            name = annotation_line.split()[0]
            # 从文件中读取图像
            jpg = Image.open(r"./VOCdevkit/VOC2007/JPEGImages" + '/' + name + ".jpg")
            png = Image.open(r"./VOCdevkit/VOC2007/SegmentationClass" + '/' + name + ".png")

            if random_data:
                jpg, png = self.get_random_data(jpg,png,(int(self.image_size[1]),int(self.image_size[0])))
            else:
                jpg, png = letterbox_image(jpg, png, (int(self.image_size[1]),int(self.image_size[0])))
            
            inputs.append(np.array(jpg)/255)
            
            # 从文件中读取图像
            png = np.array(png)
            png[png >= self.num_classes] = self.num_classes
            
            # 转化成one_hot的形式
            seg_labels = np.eye(self.num_classes+1)[png.reshape([-1])]
            seg_labels = seg_labels.reshape((int(self.image_size[1]),int(self.image_size[0]),self.num_classes+1))
            
            targets.append(seg_labels)
            i = (i + 1) % length
            if len(targets) == self.batch_size:
                tmp_inp = np.array(inputs)
                tmp_targets = np.array(targets)
                inputs = []
                targets = []
                yield tmp_inp, tmp_targets
