from re import sub
import tensorflow as tf


def get_images_ds(hparams, type_ds):
    subset = type_ds

    if subset == 'train':
        subset = 'training'
    
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=hparams.image_dir,
        validation_split = 0.2,
        subset = subset,
        seed = hparams.seed,
        image_size = hparams.input_image_sizes,
        batch_size=hparams.batch_size,
        label_mode='categorical'
    )

    #print(type_ds, " Has a shape of", ds.shape)


    return ds


def get_train_ds(hparams):
    return get_images_ds(hparams,'train')
    

def get_val_ds(hparams):
    return get_images_ds(hparams, 'validation')

def get_test_ds(hparams):
    return get_images_ds(hparams, 'test')
