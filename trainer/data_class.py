from re import sub
import tensorflow as tf
from tensorflow import keras
import keras_cv


def get_images_ds(hparams, type_ds):

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


#def get_train_ds(hparams):
#    return get_images_ds(hparams,'train')
    

def get_val_ds(hparams):
    return get_images_ds(hparams, 'validation')

def get_test_ds(hparams):
    return get_images_ds(hparams, 'test')

def get_train_ds(hparams):
    augmentation_type = hparams.model_type.lower()
    dataset = get_images_ds(hparams,'train')
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
        
        
        def augment_data(images,labels):
            inputs = {"images": images, "labels": labels}
            for layer in data_augmentation_list:
                inputs = layer(inputs)
            return inputs['images'], inputs['labels']
        
        if len(data_augmentation_list)>0:
            print('Data are augmented')
            dataset = dataset.map(augment_data)
        
    else:
        print('unsupported augmentation type %s' % (augmentation_type))
        print('model will be trained in the initial dataset')
    return dataset


