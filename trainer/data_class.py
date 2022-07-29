import tensorflow as tf


image_size = (180, 180)
batch_size = 32


def get_images_ds(hparams, type_ds):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=hparams.image_dir+ '/' +type_ds,
        validation_split=0,
        subset = type_ds,
        seed = 1337,
        image_size = hparams.image_size,
        batch_size=hparams.batch_size
    )

    return ds


def get_train_ds(hparams):
    return get_images_ds(hparams,'train')
    

def get_val_ds(hparams):
    return get_images_ds(hparams, 'validation')

def get_test_ds(hparams):
    return get_images_ds(hparams, 'test')
