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


def mobilenet_v2_model(no_class):

    # Mobilenet V2
    model = tf.keras.applications.MobileNetV2()

    # Input Size = 224 x 224 (pre-Trained model)
    my_input = model.layers[0].input

    # Removing last layer in pre-trained model (it's for 1000 classes)
    # Changes to our classe number (Our Need)
    output = model.layers[-2].output

    x = layers.Dense(1024)(output)
    x = layers.Activation('relu')(x)
    x = layers.Dense(no_class, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

