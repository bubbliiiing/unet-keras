#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from nets.unet import Unet

if __name__ == "__main__":
    model = Unet([512, 512, 3], 21)
    model.summary()

    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
