import tensorflow as tf
from keras import layers
from keras import Model


def custom_model(no_classes, img_size):
    my_input = layers.Input(shape=(img_size, img_size, 3))

    x = layers.Conv2D(32, (3, 3), activation='relu')(my_input)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(no_classes, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)


########## VGG ##########

def vgg_model(no_class, model_type):

    # VGG16
    if model_type == 'vgg16':
        model = tf.keras.applications.VGG16()
    
    # VGG19
    elif model_type == 'vgg19':
        model = tf.keras.applications.VGG19()

    # default input size is 224 x 224
    my_input = model.layers[0].input

    # Removing last layer in pre-trained model (it's for 1000 classes)
    # Changes to our classe number (Our Need)
    output = model.layers[-2].output

    x = layers.Dense(1024)(output)
    x = layers.Activation('relu')(x)
    x = layers.Dense(no_class, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)


########## MobileNet ##########

def mobilenet_model(no_class, model_type):

    # MobileNet
    if model_type == 'mobilenet':
        model = tf.keras.applications.MobileNet()
    
    # MobileNet V2
    elif model_type == 'mobilenetV2':
        model = tf.keras.applications.MobileNetV2()
    
    # MobileNet V3 Small
    elif model_type == 'mobilenetV3Small':
        model = tf.keras.applications.MobileNetV3Small()
    
    # MobileNet V3 Large
    elif model_type == 'mobilenetV3Large':
        model = tf.keras.applications.MobileNetV3Large()

    # Input Size = 224 x 224 (pr
    my_input = model.layers[0].input

    # Changes to our classe number (Our Need)
    output = model.layers[-2].output

    x = layers.Dense(1024)(output)
    x = layers.Activation('relu')(x)
    x = layers.Dense(no_class, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)


########## EfficientNet ##########

def efficientnet_model(no_class, model_type):
    
    # EfficientNetB0
    if model_type == 'efficientnetB0':
        model = tf.keras.applications.EfficientNetB0()

    # EfficientNetB1
    elif model_type == 'efficientnetB1':
        model = tf.keras.applications.EfficientNetB1()

    # EfficientNetB2
    elif model_type == 'efficientnetB2':
        model = tf.keras.applications.EfficientNetB2()
    
    # EfficientNetB3
    elif model_type == 'efficientnetB3':
        model = tf.keras.applications.EfficientNetB3()

    # EfficientNetB4
    elif model_type == 'efficientnetB4':
        model = tf.keras.applications.EfficientNetB4()

    # EfficientNetB5
    elif model_type == 'efficientnetB5':
        model = tf.keras.applications.EfficientNetB5()

    # EfficientNetB6
    elif model_type == 'efficientnetB6':
        model = tf.keras.applications.EfficientNetB6()

    # EfficientNetB7
    elif model_type == 'efficientnetB7':
        model = tf.keras.applications.EfficientNetB7()

    # Input Layer
    my_input = model.layers[0].input

    # Changes to our classe number (Our Need)
    output = model.layers[-2].output

    x = layers.Dense(1024)(output)
    x = layers.Activation('relu')(x)
    x = layers.Dense(no_class, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)


#### Xception ####

def xception_model(no_class):

    # Xception
    model = tf.keras.applications.Xception()

    # Input Size = 299 x 299 (pre-Trained model)
    my_input = model.layers[0].input

    # Changes to our classe number (Our Need)
    output = model.layers[-2].output

    x = layers.Dense(1024)(output)
    x = layers.Activation('relu')(x)
    x = layers.Dense(no_class, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)


def efficientnetV2_model(no_class, model_type):
    
    # EfficientNetV2B0
    if model_type == 'efficientnetV2B0':
        model = tf.keras.applications.EfficientNetV2B0()

    # EfficientNetV2B1
    elif model_type == 'efficientnetV2B1':
        model = tf.keras.applications.EfficientNetV2B1()

    # EfficientNetV2B2
    elif model_type == 'efficientnetV2B2':
        model = tf.keras.applications.EfficientNetV2B2()
    
    # EfficientNetV2B3
    elif model_type == 'efficientnetV2B3':
        model = tf.keras.applications.EfficientNetV2B3()

    # EfficientNetV2S
    elif model_type == 'efficientnetV2S':
        model = tf.keras.applications.EfficientNetV2S()

    # EfficientNetV2M
    elif model_type == 'efficientnetV2M':
        model = tf.keras.applications.EfficientNetV2M()

    # EfficientNetV2L
    elif model_type == 'efficientnetV2L':
        model = tf.keras.applications.EfficientNetV2L()

    # Input Layer
    my_input = model.layers[0].input

    # Changes to our classe number (Our Need)
    output = model.layers[-2].output

    x = layers.Dense(1024)(output)
    x = layers.Activation('relu')(x)
    x = layers.Dense(no_class, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)


def resnet_model(no_class, model_type):
    
    # ResNet50
    if model_type == 'resnet50':
        model = tf.keras.applications.ResNet50()

    # ResNet101
    elif model_type == 'resnet101':
        model = tf.keras.applications.ResNet101()

    # ResNet152
    elif model_type == 'resnet152':
        model = tf.keras.applications.ResNet152()
    
    # ResNet50V2
    elif model_type == 'resnet50V2':
        model = tf.keras.applications.ResNet50V2()

    # ResNet101V2
    elif model_type == 'resnet101V2':
        model = tf.keras.applications.ResNet101V2()

    # ResNet152V2
    elif model_type == 'ResNet152V2':
        model = tf.keras.applications.ResNet152V2()

    # Input Layer
    my_input = model.layers[0].input

    # Changes to our classe number (Our Need)
    output = model.layers[-2].output

    x = layers.Dense(1024)(output)
    x = layers.Activation('relu')(x)
    x = layers.Dense(no_class, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

