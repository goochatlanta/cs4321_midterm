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

def create_model(hparams):
    model_type = hparams.model_type.lower()
    if model_type == 'fully_connected':
        return create_fully_connected_model(hparams)
    elif model_type == 'no_hidden':
        return create_no_hidden(hparams)
    #elif model_type == 'cnn_model':
    #    return create_cnn_model(hparams)
    else:
        print('unsupported model type %s' % (model_type))
        return None
