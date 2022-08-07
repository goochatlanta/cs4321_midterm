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
    
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',include_top=False
    )

    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    preds = tf.keras.layers.Dense(hparams.amount_of_labels, activation='softmax')(x) #final layer with softmax activation


    model = tf.keras.Model(inputs=base_model.input, outputs=preds)


    for layer in model.layers[:-1]:
        layer.trainable = False
    

    return model

def create_model(hparams):
    model_type = hparams.model_type.lower()
    if model_type == 'mobilenetv2_frozen':
        return create_MobileNetV2_frozen(hparams)
    #elif model_type == 'cnn_model':
    #    return create_cnn_model(hparams)
    else:
        print('unsupported model type %s' % (model_type))
        return None
