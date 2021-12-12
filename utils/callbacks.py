import os

import keras
import matplotlib.pyplot as plt
import scipy.signal
from keras import backend as K


class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_dir, val_loss_flag = True):
        import datetime
        self.time_str       = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        self.save_path      = os.path.join(log_dir, "loss_" + str(self.time_str))  
        self.val_loss_flag  = val_loss_flag

        self.losses         = []
        if self.val_loss_flag:
            self.val_loss   = []
        
        os.makedirs(self.save_path)

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('loss')))
            f.write("\n")

        if self.val_loss_flag:
            self.val_loss.append(logs.get('val_loss'))
            with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
                f.write(str(logs.get('val_loss')))
                f.write("\n")

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        try:
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, 5 if len(self.losses) < 25 else 15, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
        except:
            pass

        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
            try:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, 5 if len(self.losses) < 25 else 15, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
            except:
                pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")

class ExponentDecayScheduler(keras.callbacks.Callback):
    def __init__(self,
                 decay_rate,
                 verbose=0):
        super(ExponentDecayScheduler, self).__init__()
        self.decay_rate         = decay_rate
        self.verbose            = verbose
        self.learning_rates     = []

    def on_epoch_end(self, batch, logs=None):
        learning_rate = K.get_value(self.model.optimizer.lr) * self.decay_rate
        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print('Setting learning rate to %s.' % (learning_rate))
