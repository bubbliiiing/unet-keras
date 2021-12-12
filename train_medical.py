import os

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from nets.unet import Unet
from nets.unet_training import (CE, Focal_Loss, dice_loss_with_CE,
                                dice_loss_with_Focal_Loss)
from utils.callbacks import ExponentDecayScheduler, LossHistory
from utils.dataloader_medical import UnetDataset
from utils.utils_metrics import Iou_score, f_score

'''
训练自己的语义分割模型一定需要注意以下几点：
1、该数据集是我根据网上找到的医药数据集特殊建立的训练文件，只是一个例子，用于展示数据集不是voc格式时要如何进行训练。
   
   不可以计算miou等性能指标。只用于观看医药数据集的训练效果。
   不可以计算miou等性能指标。
   不可以计算miou等性能指标。

   如果大家有自己的医药数据集需要训练，可以分为两种情况：
   a、没有标签的医药数据集：
      请按照视频里面的数据集标注教程，首先利用labelme标注图片，转换成VOC格式后利用train.py进行训练。
   b、有标签的医药数据集：
      将文件的标签格式进行转换，标签的每个像素点的值就是这个像素点所属的种类。
      因此数据集的标签需要改成，背景的像素点值为0，目标的像素点值为1。

2、训练好的权值文件保存在logs文件夹中，每个epoch都会保存一次，如果只是训练了几个step是不会保存的，epoch和step的概念要捋清楚一下。
   在训练过程中，该代码并没有设定只保存最低损失的，因此按默认参数训练完会有100个权值，如果空间不够可以自行删除。
   这个并不是保存越少越好也不是保存越多越好，有人想要都保存、有人想只保存一点，为了满足大多数的需求，还是都保存可选择性高。

3、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中

4、调参是一门蛮重要的学问，没有什么参数是一定好的，现有的参数是我测试过可以正常训练的参数，因此我会建议用现有的参数。
   但是参数本身并不是绝对的，比如随着batch的增大学习率也可以增大，效果也会好一些；过深的网络不要用太大的学习率等等。
   这些都是经验上，只能靠各位同学多查询资料和自己试试了。
'''
if __name__ == "__main__":    
    #--------------------------------------------------------------------#
    #   简单的医药分割只分背景和边缘
    #--------------------------------------------------------------------#
    num_classes     = 2
    #-------------------------------#
    #   主干网络选择
    #   vgg、resnet50
    #-------------------------------#
    backbone        = "vgg"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #   训练自己的数据集时提示维度不匹配正常，预测的东西都不一样了自然维度不匹配
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path为主干网络的权值，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，下面的Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #
    #   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
    #   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = "model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    #------------------------------#
    #   输入图片的大小
    #------------------------------#
    input_shape     = [512, 512]

    #----------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   
    #   由于resnet50中有BatchNormalization层
    #   当主干为resnet50的时候batch_size不可为1
    #----------------------------------------------------#
    #----------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #----------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2
    Freeze_lr           = 1e-4
    #----------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #----------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 2
    Unfreeze_lr         = 1e-5
    #------------------------------#
    #   数据集路径
    #------------------------------#
    VOCdevkit_path  = 'Medical_Datasets'
    #--------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #---------------------------------------------------------------------# 
    dice_loss       = False
    #---------------------------------------------------------------------# 
    #   是否使用focal loss来防止正负样本不平衡
    #---------------------------------------------------------------------# 
    focal_loss      = False
    #---------------------------------------------------------------------# 
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #---------------------------------------------------------------------# 
    cls_weights     = np.ones([num_classes], np.float32)
    #---------------------------------------------------------------------# 
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    #---------------------------------------------------------------------# 
    Freeze_Train    = True
    #---------------------------------------------------------------------# 
    #   用于设置是否使用多线程读取数据，1代表关闭多线程
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   keras里开启多线程有些时候速度反而慢了许多
    #   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    #---------------------------------------------------------------------# 
    num_workers     = 1

    #------------------------------------------------------#
    #   获取model
    #------------------------------------------------------#
    model = Unet([input_shape[0], input_shape[1], 3], num_classes, backbone)
    if model_path != '':
        #------------------------------------------------------#
        #   载入预训练权重
        #------------------------------------------------------#
        model.load_weights(model_path, by_name=True, skip_mismatch=True)

    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()

    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    logging         = TensorBoard(log_dir = 'logs/')
    checkpoint      = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}.h5',
                        monitor = 'loss', save_weights_only = True, save_best_only = False, period = 1)
    reduce_lr       = ExponentDecayScheduler(decay_rate = 0.96, verbose = 1)
    early_stopping  = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)
    loss_history    = LossHistory('logs/', val_loss_flag = False)

    if focal_loss:
        if dice_loss:
            loss = dice_loss_with_Focal_Loss(cls_weights)
        else:
            loss = Focal_Loss(cls_weights)
    else:
        if dice_loss:
            loss = dice_loss_with_CE(cls_weights)
        else:
            loss = CE(cls_weights)
            
    #------------------------------------#
    #   冻结一定部分训练
    #------------------------------------#
    if backbone == "vgg":
        freeze_layers = 17
    elif backbone == "resnet50":
        freeze_layers = 172
    else:
        raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
    #------------------------------------#
    #   冻结一定部分训练
    #------------------------------------#
    if Freeze_Train:
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
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch

        epoch_step      = len(train_lines) // batch_size
        
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        
        model.compile(loss = loss,
                optimizer = Adam(lr=lr),
                metrics = [f_score()])

        train_dataloader    = UnetDataset(train_lines, input_shape, batch_size, num_classes, True, VOCdevkit_path)

        print('Train on {} samples, with batch size {}.'.format(len(train_lines), batch_size))
        model.fit_generator(
            generator           = train_dataloader,
            steps_per_epoch     = epoch_step,
            epochs              = end_epoch,
            initial_epoch       = start_epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )
    
    if Freeze_Train:
        for i in range(freeze_layers): model.layers[i].trainable = True

    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        epoch_step      = len(train_lines) // batch_size
        
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        
        model.compile(loss = loss,
                optimizer = Adam(lr=lr),
                metrics = [f_score()])
                
        train_dataloader    = UnetDataset(train_lines, input_shape, batch_size, num_classes, True, VOCdevkit_path)

        print('Train on {} samples, with batch size {}.'.format(len(train_lines), batch_size))
        model.fit_generator(
            generator           = train_dataloader,
            steps_per_epoch     = epoch_step,
            epochs              = end_epoch,
            initial_epoch       = start_epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )
    
