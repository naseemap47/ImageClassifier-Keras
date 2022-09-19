from keras import layers
from keras import Model


def Model(no_classes, img_size):
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
