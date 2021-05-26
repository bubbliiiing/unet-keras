import os

from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.optimizers import Adam

from nets.unet import Unet
from nets.unet_training import LossHistory
from nets.unet_training_medical import CE, Generator, dice_loss_with_CE
from utils.metrics import Iou_score, f_score

'''
需要注意的是，这个医药数据集的训练只是一个例子，用于展示数据集不是voc格式时要如何进行训练。
所以这个文件只是根据我网上找到的医药数据集特殊建立的训练文件。只用于观看医药数据集的训练效果。
不可以计算miou等性能指标。

如果大家有自己的医药数据集需要标注后训练，完全可以按照视频里面的教程
首先利用labelme标注图片，转换称VOC格式后利用train.py进行训练。
训练自己标注的医药数据集的步骤和正常数据集的步骤一摸一样，不需要用到这个train_medical.py！
'''
if __name__ == "__main__":    
    log_dir = "logs/"
    #------------------------------#
    #   输入图片的大小
    #------------------------------#
    inputs_size = [512,512,3]
    #--------------------------------------------------------------------#
    #   简单的医药分割只分背景和边缘
    #--------------------------------------------------------------------#
    num_classes = 2
    #--------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #---------------------------------------------------------------------# 
    dice_loss = True
    #------------------------------#
    #   数据集路径
    #------------------------------#
    dataset_path = "Medical_Datasets/"

    # 获取model
    model = Unet(inputs_size,num_classes)
    
    #-------------------------------------------#
    #   权值文件的下载请看README
    #   权值和主干特征提取网络一定要对应
    #-------------------------------------------#
    model_path = "./model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    with open(os.path.join(dataset_path, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()

    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}.h5',
                                    monitor='loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=12, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir)
    loss_history = LossHistory(log_dir)

    freeze_layers = 17
    for i in range(freeze_layers): model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        lr              = 1e-4
        Init_Epoch      = 0
        Freeze_Epoch    = 50
        Batch_size      = 2

        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])

        gen             = Generator(Batch_size, train_lines, inputs_size, num_classes, dataset_path).generate()
        
        epoch_size      = len(train_lines) // Batch_size

        if epoch_size == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, with batch size {}.'.format(len(train_lines), Batch_size))
        model.fit_generator(gen,
                steps_per_epoch=epoch_size,
                epochs=Freeze_Epoch,
                initial_epoch=Init_Epoch,
                callbacks=[checkpoint_period, reduce_lr,tensorboard])
    
    
    for i in range(freeze_layers): model.layers[i].trainable = True

    if True:
        lr              = 1e-5
        Freeze_Epoch    = 50
        Unfreeze_Epoch  = 100
        Batch_size      = 2

        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])

        gen             = Generator(Batch_size, train_lines, inputs_size, num_classes, dataset_path).generate()
        
        epoch_size      = len(train_lines) // Batch_size

        if epoch_size == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, with batch size {}.'.format(len(train_lines), Batch_size))
        model.fit_generator(gen,
                steps_per_epoch=epoch_size,
                epochs=Unfreeze_Epoch,
                initial_epoch=Freeze_Epoch,
                callbacks=[checkpoint_period, reduce_lr,tensorboard])
