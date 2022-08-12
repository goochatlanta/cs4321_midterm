from re import sub
import tensorflow as tf
from tensorflow import keras
import keras_cv
"""
Data_module
Description: This module contains functions to get the data from a given directory.
             The data are not automatically split into different datasets but 
             there must be one directory for each dataset. It also contains functions 
             for data augmentation outside the model. The trained data are augmented 
             if the right arguments are passed
Data augmentation: random_augmentation, random_flip, MixUp
"""
def get_images_ds(hparams, type_ds):
    """
    Inputs: 
      hparams: <arg object> the arguments file
      type_ds: <string> the type of the dataset to load
    Outputs:
      Returns the dataset
    """
    if type_ds == 'train':
    
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=hparams.train_dir,
            validation_split = None,
            seed = hparams.seed,
            image_size = hparams.input_image_sizes,
            batch_size=hparams.batch_size,
            label_mode='categorical'
        )
    elif type_ds == 'test':

        ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=hparams.test_dir,
            validation_split = None,
            seed = hparams.seed,
            batch_size = hparams.batch_size,
            image_size = hparams.input_image_sizes,
            label_mode='categorical'
        )
    elif type_ds == 'validation':
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=hparams.val_dir,
            validation_split = None,
            seed = hparams.seed,
            image_size = hparams.input_image_sizes,
            batch_size=hparams.batch_size,
            label_mode='categorical'
        )
    
    return ds

    
def get_val_ds(hparams):
    """
    Inputs: 
      hparams: <arg object> the arguments file
    
    Outputs:
      Returns the validetion dataset    
    """
    return get_images_ds(hparams, 'validation')

def get_test_ds(hparams):
    """
    Inputs: 
      hparams: <arg object> the arguments file
    
    Outputs:
      Returns the testing dataset
    """
    return get_images_ds(hparams, 'test')

def get_train_ds(hparams):
    """
    Inputs: 
      hparams: <arg object> the arguments file
    
    Outputs:
      Returns the training dataset. if an data augmentation has been passed
      It returns the augmented data.
    """
    augmentation_type = hparams.model_type.lower()
    dataset = get_images_ds(hparams,'train')
    # If the the dataset is for a tsne use no augmentation will take place
    if hparams.tsne_ds:
        return dataset

    # create a list of layers that contain the type of the augmentaion
    if hparams.data_augmentation_list:
        data_augmentation_list = []

        if 'random_augmentation' in hparams.data_augmentation_list:
            data_augmentation_list.append(
                keras_cv.layers.RandAugment(
                    value_range=(0, 255), augmentations_per_image=3, magnitude=0.5)
            )
        if 'random_flip' in hparams.data_augmentation_list:
            data_augmentation_list.append(
                keras_cv.layers.RandomFlip()
            )
        if 'MixUp' in hparams.data_augmentation_list:
            data_augmentation_list.append(
                keras_cv.layers.MixUp(alpha=0.3)
            )
        
        # helping function that will augment the ds. here is where thae augmentation part 
        # takes pplace
        def augment_data(images,labels):
            inputs = {"images": images, "labels": labels}
            for layer in data_augmentation_list:
                inputs = layer(inputs)
            return inputs['images'], inputs['labels']
        
        # map the augmented data
        if len(data_augmentation_list)>0:
            print('Data are augmented')
            dataset = dataset.map(augment_data)
        
    else:
        print('unsupported augmentation type %s' % (augmentation_type))
        print('model will be trained in the initial dataset')
    return dataset


