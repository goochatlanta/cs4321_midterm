# Copyright 2020, Prof. Marko Orescanin, NPS
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Created by marko.orescanin@nps.edu on 7/21/20


import os
import sys
import yaml

import tensorflow as tf
import params
import models
import optimizers
import numpy as np
import pickle
import callbacks



def main():

    hparams = params.get_hparams()

    if not os.path.exists(hparams.model_dir):
        os.mkdir(hparams.model_dir)

    params.save_hparams(hparams)

    # import data

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # %% make one_hot_encoded
    trainY = tf.keras.utils.to_categorical(y_train)
    testY = tf.keras.utils.to_categorical(y_test)

    # Lets look at few data samples
    print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

    model = models.create_model(hparams)

    model.summary()

    model.compile(optimizer=optimizers.get_optimizer(hparams),
                                   loss=hparams.loss_type,
                                   metrics=[hparams.eval_metrics])

    print(model.summary())

    history = model.fit(x_train,trainY,
                                epochs=hparams.num_epochs,
                                validation_split=0.2,
                                callbacks=callbacks.make_callbacks(hparams))

    with open(os.path.join(hparams.model_dir, "history.pickle"), 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == "__main__":
    main()
