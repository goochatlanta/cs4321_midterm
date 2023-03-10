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

"""params.py

This module contains all parameter handling for the project, including
all command line tunable parameter definitions, any preprocessing
of those parameters, or generation/specification of other parameters.

"""
from scipy.constants import golden, pi
import tensorflow as tf
import numpy as np
import argparse
import os
import datetime
import yaml


def make_argparser():
    parser = argparse.ArgumentParser(description='Arguments to run training for our CS4321 MidTerm')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, default='/data/cs4321/KCAteam/data/test')
    parser.add_argument('--train_dir', type=str, default='/data/cs4321/KCAteam/data/train')
    parser.add_argument('--val_dir', type=str, default='/data/cs4321/KCAteam/data/validation')
    parser.add_argument('--image_dir', type=str, default='/data/cs4321/KCAteam/data/merged')
    parser.add_argument('--model_type', type=str, default='fully_connected',
                        help="the type of model to use. allowed inputs are fully_connected and cnn")
    parser.add_argument('--regression', type=str2bool, default=False,
                        help="is it a regression model")
    parser.add_argument('--input_image_sizes', type=int, default=(299, 299),
                        help="The size of the images")
    parser.add_argument('--continue_training', type=str2bool, default=False,
                        help="continue training from a checkpoint")
    parser.add_argument('--checkpoint_path', type=str,
                        help="path to the checkpoint to continue training")
    parser.add_argument('--predict', type=str2bool, default=False,
                        help="predict from a checkpoint, use checkpoint flag to pass a model")
    parser.add_argument('--seed', type=int, default=1337,
                        help="the seed")
    parser.add_argument('--num_epochs', type=int, default=2,
                        help="the number of epochs")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="The batch size to use in the training")
    parser.add_argument('--amount_of_labels', type=int, default=8,
                        help="The amount of lalbels on the last layer")
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="specify the optimizer for the model")
    parser.add_argument("--callback_list", type=str, default=None,
                        help="the callbacks to be added")
    parser.add_argument("--base_learning_rate", type=int, default=0.001,
                        help="specify the base learning rate for the specified optimizer for the model")
    # how many epochs do we want the learning rate not be changed
    parser.add_argument("--constant_learning_rate_epochs", type=int, default=40,
                        help="specify how many epoch the learning rate will remain constant and not decrease")
    parser.add_argument("--loss_type", type=str, default="categorical_crossentropy",
                        help=" loss type: Options [categorical_crossentropy | binary_crossentropy]")
    parser.add_argument("--eval_metrics", type=str, default=None,
                        help="a list of the metrics of interest, seperated by commas")
    # multi-gpu training arguments
    parser.add_argument('--mgpu_run', type=str2bool, default=False,
                        help="multi gpu run")
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="number of gpu's on the machine, juno is 2")
    # multi-processing arguments
    parser.add_argument('--use_multiprocessing', type=str2bool, default=True,
                        help="specifys weather to use use_multiprocessing in .fit_genrator method ")
    parser.add_argument("--workers", type=int, default=6,
                        help="number of CPU's, for my machine 6 workers, for Juno 18")
    # data augmentation arguments
    parser.add_argument('--data_augmentation_list', type=str, default=None,
                        help='specify the data augmentation type. it can be more than one')
    # argument to run from the beginning the unfozen model
    parser.add_argument('--only_fine_tuning', type=str, default=None,
                        help='specify the folder that contains the model to test')
    # how many epochs to train the unfrozen model
    parser.add_argument('--num_fine_epochs', type=int, default=40,
                        help="the number of fine tunning epochs")
    # number of layers to be unfrozen
    parser.add_argument('--unfrozen_layers', type=int, default=70,
                        help="how many layers to be unfrozen")
    # the length of the dense layer before the classifier 
    parser.add_argument('--length_of_dense_layers' ,type=str, default="256",
                        help="Specifiy the lengths of the dense laeyers before the last softmax")
    # argument to continue trainin
    parser.add_argument('--con_fine_tunning' ,type=str, default=None,
                        help="If you want to keep train the same model")
    # argument to Know if we want to load the dataset for the tsne. no augmentation will be added
    parser.add_argument('--tsne_ds', type=str,
                        help="Only for the report")
    return parser.parse_args()


# you have to use str2bool
# because of an issue with argparser and bool type
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1'):
        return True
    elif v.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_hparams():
    """any preprocessing, special handling of the hparams object"""

    parser = make_argparser()
    print(parser)

    return parser


def save_hparams(hparams):
    path_ = os.path.join(hparams.model_dir, 'params.txt')
    hparams_ = vars(hparams)
    with open(path_, 'w') as f:
        for arg in hparams_:
            print(arg, ':', hparams_[arg])
            f.write(arg + ':' + str(hparams_[arg]) + '\n')

    path_ = os.path.join(hparams.model_dir, 'params.yml')
    with open(path_, 'w') as f:
        yaml.dump(hparams_, f,
                  default_flow_style=False)  # save hparams as a yaml, since that's easier to read and use


