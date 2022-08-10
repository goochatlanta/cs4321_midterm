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


def create_model_from_app(hparams, model_type):

    IMG_SHAPE = hparams.input_image_sizes + (3,)

    if (model_type == 'mobilenetv2'):
        app = tf.keras.applications.MobileNetV2
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif (model_type == 'vgg16'):
        app = tf.keras.applications.VGG16
        preprocess = tf.keras.applications.vgg16.preprocess_input
    elif (model_type == 'resnet50'):
        app = tf.keras.applications.ResNet50
        preprocess = tf.keras.applications.resnet50.preprocess_input
    
    
    base_model = app(
        weights='imagenet',include_top=False,input_shape=IMG_SHAPE
    )

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = preprocess(inputs)
    x = base_model(x,training=False)
    #x = tf.keras.layers.Conv2D(32,activation='relu', kernel_size =(3,3))(x)
    #x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #x = tf.keras.layers.Flatten()(x)
    for len in hparams.length_of_dense_layers:
        x = tf.keras.layers.Dense(len, activation='relu')(x)
    #x = tf.keras.layers.Dropout(.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    preds = tf.keras.layers.Dense(hparams.amount_of_labels, activation='softmax')(x) #final layer with softmax activation
    model = tf.keras.Model(inputs=inputs, outputs=preds)


    for layer in model.layers[:-2]:
        layer.trainable = False
    
    model.summary()
    return model

def create_model(hparams):
    model_type = hparams.model_type.lower()
    if model_type in ['mobilenetv2' ,'resnet50', 'vgg16']:
        return create_model_from_app(hparams, model_type)
    #elif model_type == 'cnn_model':
    #    return create_cnn_model(hparams)
    else:
        print('unsupported model type %s' % (model_type))
        return None

def unfreeze_model(hparams,model):
    for layer in model.layers:
        if hparams.model_type.lower() in layer.name.lower():
            for base_layer in layer.layers[-hparams.unfrozen_layers:]:
                base_layer.trainable = True
    return model

