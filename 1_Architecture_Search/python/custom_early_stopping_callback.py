import numpy as np
from scipy.stats import norm
import keras
import constants


class CustomEarlyStoppingCallback(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, mean_std_list, percentile_threshold):
        super(CustomEarlyStoppingCallback, self).__init__()
        self.mean_std_list = mean_std_list
        self.percentile_threshold = percentile_threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get(constants.val_acc_name)
        mu, std = self.mean_std_list[epoch]
        if percentile(val_acc, mu, std) < self.percentile_threshold[epoch]:
            self.model.stop_training = True


def percentile(x, mu, std):
    p = norm.cdf(x, mu, std)
    return p
