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
from tabnanny import verbose
import yaml
import glob

import tensorflow as tf
import params
import models
import data_class
import optimizers
import numpy as np
import pickle
import callbacks
import report


def main():
    # prepare data for trainng
    hparams = params.get_hparams()
    train_ds = data_class.get_train_ds(hparams)
    val_ds = data_class.get_val_ds(hparams)
    
    # it will continue the training of an unfreezed existing model
    if not hparams.con_fine_tunning:
        # it will unfreeze and continue the training of an existing model
        if not hparams.only_fine_tuning:

            # create model_dir and save the parameters
            if not os.path.exists(hparams.model_dir):
                os.mkdir(hparams.model_dir)
            params.save_hparams(hparams)

            # Build and compile the model to train
            model = models.create_model(hparams)
            model.compile(optimizer=optimizers.get_optimizer(hparams),
                                           loss=hparams.loss_type,
                                           metrics=[hparams.eval_metrics])
            #Train the model
            history = model.fit(train_ds,
                                epochs=hparams.num_epochs,
                                validation_data = val_ds,
                                callbacks=callbacks.make_callbacks(hparams),
                                verbose=2)
            # save the history of the model
            with open(os.path.join(hparams.model_dir, "history.pickle"), 'wb') as f:
                pickle.dump(history.history, f)

            # save the best model
            list_of_checkpoints = glob.glob(f'{hparams.model_dir}*checkpoint*')
            latest_check = max(list_of_checkpoints, key=os.path.getctime)
            model = tf.keras.models.load_model(latest_check)
            model.save(hparams.model_dir+'/'+hparams.model_type.lower())
            print('report for freezed model')
            report.test_model(hparams,model)
            
            # starts the fine_tuning training
            fine_tuning(hparams, train_ds, val_ds, model)
            report.test_model(hparams,model)

        else:
            # load the freezed model and starts fine tunning
            print('load the frozen model')
            model = tf.keras.models.load_model(hparams.model_dir+hparams.model_type.lower())
            fine_tuning(hparams, train_ds, val_ds, model, False)
            report.test_model(hparams, model)
    else:
        # load the unfreezed model and continius the training
        print('load the unfrozen model')
        model = tf.keras.models.load_model(hparams.model_dir+hparams.model_type.lower()+'_unfrozen')
        fine_tuning(hparams, train_ds, val_ds, model)
        report.test_model(hparams, model)

def fine_tuning(hparams, train_ds, val_ds, model, compile = True):
    """
    Inputs: 
      hparams: <arg object> the arguments file
      train_ds: <Dataset> the trained dataset
      val_ds: <Dataset> the validation dataset
      model: <Keras.Model> the model for fine tuning
      compile: <boolean> True if we want to compile the model
    Outputs:
      Returns the dataset
    """

    # unfreeze the model
    model=models.unfreeze_model(hparams, model)
    # change the optimizer
    hparams.optimizer = 'SGD'
    if compile:
        model.compile(optimizer=optimizers.get_optimizer(hparams),
                                       loss=hparams.loss_type,
                                       metrics=[hparams.eval_metrics])
    # train the unfreezed model
    history_fine = model.fit(train_ds,
                        epochs=(hparams.num_epochs+hparams.num_fine_epochs),
                        initial_epoch = hparams.num_epochs,
                        validation_data = val_ds,
                        callbacks=callbacks.make_callbacks(hparams),
                        verbose=2)
    # save history of fine tuning
    with open(os.path.join(hparams.model_dir, "history_fine.pickle"), 'wb') as f:
        pickle.dump(history_fine.history, f)
    # save the best model
    list_of_checkpoints = glob.glob(f'{hparams.model_dir}*checkpoint*')
    latest_check = max(list_of_checkpoints, key=os.path.getctime)
    model = tf.keras.models.load_model(latest_check)
    model.save(hparams.model_dir+hparams.model_type.lower()+'_unfrozen')


if __name__ == "__main__":
    main()
