import os
import scipy.io
import tensorflow as tf

def get_optimizer(hparams):
    """
    Function to get optimizer string and return the corresponding TF optimizer
    :param opt_string: string optimizer
    :param lr: learning rate
    :param name: name for optimizer
    :return: TF optimizer
    """
    if hparams.optimizer == "SGD":
        #the learning rate is the half from what is passed
        print("SGD lr/2")
        return tf.keras.optimizers.SGD(lr=(hparams.base_learning_rate/2), decay=0.0, momentum=0.9, nesterov=False)
    elif hparams.optimizer == "adam":
        return tf.keras.optimizers.Adam(lr =(hparams.base_learning_rate) )
    else:
        raise ValueError("""optimizer not defined in "get_optimizer" function.""")

