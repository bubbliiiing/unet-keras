import datetime
import os

import keras.backend as K
import numpy as np
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard)
from keras.layers import Conv2D, Dense, DepthwiseConv2D
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

from nets.unet import Unet
from nets.unet_training import (CE, Focal_Loss, dice_loss_with_CE,
                                dice_loss_with_Focal_Loss, get_lr_scheduler)
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
    #   
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   如果一定要从0开始，可以了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = "model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    #------------------------------#
    #   输入图片的大小
    #------------------------------#
    input_shape     = [512, 512]

    #----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，此时仅仅进行冻结训练。
    #      
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练权重开始训练： 
    #       Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True（默认参数）
    #       Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False（不冻结训练）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。optimizer_type = 'sgd'，Init_lr = 1e-2。
    #   （二）从主干网络的预训练权重开始训练：
    #       Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 300，Freeze_Train = True（冻结训练）
    #       Init_Epoch = 0，UnFreeze_Epoch = 300，Freeze_Train = False（不冻结训练）
    #       其中：由于从主干网络的预训练权重开始训练，主干的权值不一定适合目标检测，需要更多的训练跳出局部最优解。
    #             UnFreeze_Epoch可以在200-300之间调整，YOLOV5和YOLOX均推荐使用300。optimizer_type = 'sgd'，Init_lr = 1e-2。
    #   （三）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2
    #------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 2
    #------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    #                   如果设置Freeze_Train=False，建议使用优化器为sgd
    #------------------------------------------------------------------#
    Freeze_Train        = True

    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    #------------------------------------------------------------------#
    save_period         = 1
    
    #-------------------------------#
    #   数据集路径
    #-------------------------------#
    VOCdevkit_path  = 'Medical_Datasets'
    #------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #------------------------------------------------------------------#
    dice_loss       = False
    #------------------------------------------------------------------#
    #   是否使用focal loss来防止正负样本不平衡
    #------------------------------------------------------------------#
    focal_loss      = False
    #------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    #------------------------------------------------------------------#
    cls_weights     = np.ones([num_classes], np.float32)
    #-------------------------------------------------------------------#
    #   用于设置是否使用多线程读取数据，1代表关闭多线程
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    #-------------------------------------------------------------------#
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
            
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    num_train   = len(train_lines)
        
    for layer in model.layers:
        if isinstance(layer, DepthwiseConv2D):
                layer.add_loss(l2(weight_decay)(layer.depthwise_kernel))
        elif isinstance(layer, Conv2D) or isinstance(layer, Dense):
                layer.add_loss(l2(weight_decay)(layer.kernel))
                
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        if Freeze_Train:
            #------------------------------------#
            #   冻结一定部分训练
            #------------------------------------#
            if backbone == "vgg":
                freeze_layers = 17
            elif backbone == "resnet50":
                freeze_layers = 172
            else:
                raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
            for i in range(freeze_layers): model.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

        #-------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size  = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch
        
        #-------------------------------------------------------------------#
        #   判断当前batch_size与64的差别，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs     = 64
        Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
        Min_lr_fit  = max(batch_size / nbs * Min_lr, 1e-6)

        optimizer = {
            'adam'  : Adam(lr = Init_lr_fit, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        model.compile(loss = loss,
                optimizer = optimizer,
                metrics = [f_score()])
    
        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step          = num_train // batch_size

        if epoch_step == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        train_dataloader    = UnetDataset(train_lines, input_shape, batch_size, num_classes, True, VOCdevkit_path)

        #-------------------------------------------------------------------------------#
        #   训练参数的设置
        #   logging         用于设置tensorboard的保存地址
        #   checkpoint      用于设置权值保存的细节，period用于修改多少epoch保存一次
        #   lr_scheduler       用于设置学习率下降的方式
        #   early_stopping  用于设定早停，loss多次不下降自动结束训练，表示模型基本收敛
        #-------------------------------------------------------------------------------#
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join('logs', "loss_" + str(time_str))
        logging         = TensorBoard(log_dir)
        loss_history    = LossHistory(log_dir, val_loss_flag=False)
        checkpoint      = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}.h5',
                                monitor = 'loss', save_weights_only = True, save_best_only = False, period = save_period)
        early_stopping  = EarlyStopping(monitor='loss', min_delta = 0, patience = 10, verbose = 1)
        lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
        callbacks       = [logging, loss_history, checkpoint, lr_scheduler]

        if start_epoch < end_epoch:
            model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if num_workers > 1 else False,
                workers             = num_workers,
                callbacks           = callbacks
            )
        #---------------------------------------#
        #   如果模型有冻结学习部分
        #   则解冻，并设置参数
        #---------------------------------------#
        if Freeze_Train:
            batch_size  = Unfreeze_batch_size
            start_epoch = Freeze_Epoch if start_epoch < Freeze_Epoch else start_epoch
            end_epoch   = UnFreeze_Epoch
                
            #-------------------------------------------------------------------#
            #   判断当前batch_size与64的差别，自适应调整学习率
            #-------------------------------------------------------------------#
            nbs     = 64
            Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
            Min_lr_fit  = max(batch_size / nbs * Min_lr, 1e-6)
            #---------------------------------------#
            #   获得学习率下降的公式
            #---------------------------------------#
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
            lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
            callbacks       = [logging, loss_history, checkpoint, lr_scheduler]
            
            for i in range(len(model.layers)): 
                model.layers[i].trainable = True
            model.compile(loss = loss,
                    optimizer = optimizer,
                    metrics = [f_score()])

            epoch_step      = num_train // batch_size

            if epoch_step == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            train_dataloader.batch_size    = Unfreeze_batch_size

            model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if num_workers > 1 else False,
                workers             = num_workers,
                callbacks           = callbacks
            )
