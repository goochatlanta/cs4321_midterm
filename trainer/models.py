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
    """
    Inputs: 
      hparams: <arg object> the arguments file
      model_type: <string> the type of the base model
    Outputs:
      Returns the architecture of the model. The model is frozen
    Description: This function creates a model that contains a base model
      from the kers.aplications module and adds a new architecture on the 
      top of it. 
    Base models: mobilenetv2, vgg16, resnet50
    """
    # constant for the input model
    IMG_SHAPE = hparams.input_image_sizes + (3,)

    # load the base mode
    if (model_type == 'mobilenetv2'):
        app = tf.keras.applications.MobileNetV2
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif (model_type == 'vgg16'):
        app = tf.keras.applications.VGG16
        preprocess = tf.keras.applications.vgg16.preprocess_input
    elif (model_type == 'resnet50'):
        app = tf.keras.applications.ResNet50
        preprocess = tf.keras.applications.resnet50.preprocess_input
    
    # Build the model
    # remove the classifier of the loaded model
    base_model = app(
        weights='imagenet',include_top=False,input_shape=IMG_SHAPE
    )
    # Add preprocessed layers to feed the base model 
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = preprocess(inputs)
    x = base_model(x,training=False)
    # Add a 1 dimenssion layer on the top of the base model
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    for len in hparams.length_of_dense_layers:
        x = tf.keras.layers.Dense(len, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    # Add the classifier
    preds = tf.keras.layers.Dense(hparams.amount_of_labels, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=preds)

    # Freeze the model except our layers
    for layer in model.layers[:-3]:
        layer.trainable = False
    model.summary()
    return model

def create_model(hparams):
    model_type = hparams.model_type.lower()
    if model_type in ['mobilenetv2' ,'resnet50', 'vgg16']:
        return create_model_from_app(hparams, model_type)
    else:
        print('unsupported model type %s' % (model_type))
        return None

def unfreeze_model(hparams,model):
    """
    Inputs: 
      hparams: <arg object> the arguments file
      model: <keras.model> The model that we want to unfreeze
    Outputs:
      Returns the unfrozen model. The layers that we want unfreeze are 
      passed in the arguments
    """
    for layer in model.layers:
        if hparams.model_type.lower() in layer.name.lower():
            for base_layer in layer.layers[-hparams.unfrozen_layers:]:
                base_layer.trainable = True
            # unfreeze the first 10 layers
            for base_layer in layer.layers[:10]:
                base_layer.trainable = True
            print('10 first layers unfrozen')
    return model

