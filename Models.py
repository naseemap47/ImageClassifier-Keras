from keras.applications import MobileNetV2
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


def mobilenet_v2_model(no_classes, img_size=224):
    input_tensor = layers.Input(shape=(img_size, img_size, 3))

    base_model = MobileNetV2(
        input_tensor==input_tensor,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
        classes=no_classes,
        classifier_activation="softmax"
    )

    x = base_model.output
    x = layers.Dense(1024, activation='relu')(x)
    predictions = layers.Dense(no_classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)

