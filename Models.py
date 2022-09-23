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


########### MobileNet ##########

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

    # Input Size = 224 x 224 (pre-Trained model)
    my_input = model.layers[0].input

    # Removing last layer in pre-trained model (it's for 1000 classes)
    # Changes to our classe number (Our Need)
    output = model.layers[-2].output

    x = layers.Dense(1024)(output)
    x = layers.Activation('relu')(x)
    x = layers.Dense(no_class, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

