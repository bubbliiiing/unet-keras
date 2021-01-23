import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from unet import Unet


class miou_Unet(Unet):
    def detect_image(self, image):
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        img, nw, nh = self.letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))
        img = np.asarray([np.array(img)/255])

        pr = self.model.predict(img)[0]
        pr = pr.argmax(axis=-1).reshape([self.model_image_size[0],self.model_image_size[1]])
        pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]
        
        image = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h), Image.NEAREST)
        return image

unet = miou_Unet()

image_ids = open("VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt",'r').read().splitlines()

if not os.path.exists("miou_pr_dir"):
    os.makedirs("miou_pr_dir")

for image_id in tqdm(image_ids):
    image_path = "VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    image = unet.detect_image(image)
    image.save("miou_pr_dir/" + image_id + ".png")
