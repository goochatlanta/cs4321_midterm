import tensorflow as tf

def create_fully_connected_model(hparams):
    model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=hparams.input_shape),
          tf.keras.layers.Dense(200, activation='sigmoid'),
          tf.keras.layers.Dense(60, activation='sigmoid'),
          tf.keras.layers.Dense(10, activation='softmax')
        ])
    return model


def create_no_hidden(hparams):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=hparams.input_shape),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


def create_MobileNetV2_frozen(hparams):

    IMG_SHAPE = hparams.input_image_sizes + (3,)
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',include_top=False,input_shape=IMG_SHAPE
    )

    inputs = tf.keras.Input(shape=(299, 299, 3))
    x= tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    preds = tf.keras.layers.Dense(hparams.amount_of_labels, activation='softmax')(x) #final layer with softmax activation
    model = tf.keras.Model(inputs=base_model.input, outputs=preds)


    for layer in model.layers[:-2]:
        layer.trainable = False
    

    return model

def create_model(hparams):
    model_type = hparams.model_type.lower()
    if model_type == 'mobilenetv2':
        return create_MobileNetV2_frozen(hparams)
    #elif model_type == 'cnn_model':
    #    return create_cnn_model(hparams)
    else:
        print('unsupported model type %s' % (model_type))
        return None

def unfreeze_model(hparams,model):
    for layer in model.layers[-hparams.unfrozen_layers:]:
        layer.trainable = True
    return model