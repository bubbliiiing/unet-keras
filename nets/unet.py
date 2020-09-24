import numpy as np
from keras.models import *
from keras.layers import *
from nets.vgg16 import VGG16

def Unet(input_shape=(256,256,3), num_classes=21):
    inputs = Input(input_shape)
    feat1, feat2, feat3, feat4, feat5 = VGG16(inputs) 
      
    channels = [64, 128, 256, 512]

    P5_up = UpSampling2D(size=(2, 2))(feat5)
    P4 = Concatenate(axis=3)([feat4, P5_up])
    P4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P4)
    P4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P4)

    P4_up = UpSampling2D(size=(2, 2))(P4)
    P3 = Concatenate(axis=3)([feat3, P4_up])
    P3 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P3)
    P3 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P3)

    P3_up = UpSampling2D(size=(2, 2))(P3)
    P2 = Concatenate(axis=3)([feat2, P3_up])
    P2 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P2)
    P2 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P2)

    P2_up = UpSampling2D(size=(2, 2))(P2)
    P1 = Concatenate(axis=3)([feat1, P2_up])
    P1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P1)
    P1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P1)

    P1 = Conv2D(num_classes, 1, activation="softmax")(P1)

    model = Model(inputs=inputs, outputs=P1)
    return model

