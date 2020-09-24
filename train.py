import time
import keras
import numpy as np
from nets.unet import Unet
from nets.unet_training import Generator, dice_loss_with_CE, CE
from keras.utils.data_utils import get_file
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.metrics import categorical_accuracy
from keras import backend as K
from PIL import Image
from utils.metrics import Iou_score, f_score


if __name__ == "__main__":    
    inputs_size = [512,512,3]
    log_dir = "logs/"
    #---------------------#
    #   分类个数+1
    #   2+1
    #---------------------#
    num_classes = 21
    #--------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #---------------------------------------------------------------------# 
    dice_loss = False

    # 获取model
    model = Unet(inputs_size,num_classes)
    # model.summary()

    model_path = "./model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # 打开数据集的txt
    with open(r"VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt","r") as f:
        train_lines = f.readlines()

    # 打开数据集的txt
    with open(r"VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt","r") as f:
        val_lines = f.readlines()
        
    # 保存的方式，3世代保存一次
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=12, verbose=1)
    # tensorboard
    tensorboard = TensorBoard(log_dir=log_dir)

    freeze_layers = 17

    for i in range(freeze_layers): model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

    if True:
        lr = 1e-4
        Init_Epoch = 0
        Freeze_Epoch = 50
        Batch_size = 4
        # 交叉熵
        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))

        gen = Generator(Batch_size, train_lines, inputs_size, num_classes).generate()
        gen_val = Generator(Batch_size, val_lines, inputs_size, num_classes).generate(False)
        # 开始训练
        model.fit_generator(gen,
                steps_per_epoch=max(1, len(train_lines)//Batch_size),
                validation_data=gen_val,
                validation_steps=max(1, len(val_lines)//Batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[checkpoint_period, reduce_lr,tensorboard])
    
    
    for i in range(freeze_layers): model.layers[i].trainable = True

    if True:
        lr = 1e-5
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100
        Batch_size = 2
        # 交叉熵
        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))

        gen = Generator(Batch_size, train_lines, inputs_size, num_classes).generate()
        gen_val = Generator(Batch_size, val_lines, inputs_size, num_classes).generate(False)
        # 开始训练
        model.fit_generator(gen,
                steps_per_epoch=max(1, len(train_lines)//Batch_size),
                validation_data=gen_val,
                validation_steps=max(1, len(val_lines)//Batch_size),
                epochs=100,
                initial_epoch=50,
                callbacks=[checkpoint_period, reduce_lr,tensorboard])
