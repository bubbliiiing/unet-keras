#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from unet import Unet
from PIL import Image

unet = Unet()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:         
        print('Open Error! Try again!')
        continue
    else:
        r_image = unet.detect_image(image)
        r_image.show()
