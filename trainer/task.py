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
import data_class
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
    train_ds = data_class.get_train_ds(hparams)
    val_ds = data_class.get_val_ds(hparams)
    test_ds = data_class.get_test_ds(hparams)
    

    #Generate the model to train
    model = models.create_model(hparams)

    model.summary()

    model.compile(optimizer=optimizers.get_optimizer(hparams),
                                   loss=hparams.loss_type,
                                   metrics=[hparams.eval_metrics])

    print(model.summary())


    #Train the model

    history = model.fit(train_ds,
                        epochs=hparams.num_epochs,
                        validation_data = val_ds,
                                callbacks=callbacks.make_callbacks(hparams))

    with open(os.path.join(hparams.model_dir, "history.pickle"), 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == "__main__":
    main()
