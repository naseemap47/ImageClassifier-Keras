from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import cv2
import numpy as np
import os


f = open('classes.txt')
class_names = f.read().splitlines()

def data_to_list(path_to_data, img_size):
    images = []
    class_no = []
    list_class = os.listdir(path_to_data)
    num_class = len(list_class)
    for x in range(0, num_class):
        img_list = os.listdir(os.path.join(path_to_data, str(x)))
        for y in img_list:
            try:
                img = cv2.imread(os.path.join(path_to_data, str(x), y))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                class_no.append(x)
            except:
                print(f'[INFO] Failed to Read {os.path.join(path_to_data, str(x), y)} Image')
                continue

        print(f'[INFO] Extracted Class: {class_names[x]}')
    images = np.array(images)
    class_no = np.array(class_no)
    return images, class_no, num_class


def create_generators(batch_size, no_class,
                      x_train, y_train,
                      x_val, y_val,
                      model_type):
    # to_categorical
    y_train = to_categorical(y_train, no_class)
    y_val = to_categorical(y_val, no_class)

    ###### Preprocessor ######
    # Custom
    if model_type == 'custom':
        train_preprocessor = ImageDataGenerator(
            rescale=1/255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        val_preprocessor = ImageDataGenerator(rescale=1/255)
    
    # VGG
    elif model_type == 'vgg16':
        train_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
            # rotation_range=10,
            # width_shift_range=0.1,
            # height_shift_range=0.1
        )
        val_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input
        )
    elif model_type == 'vgg19':
        train_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
        )
        val_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg19.preprocess_input
        )

    # MobileNet
    elif model_type == 'mobilenet':
        train_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
            # rotation_range=10,
            # width_shift_range=0.1,
            # height_shift_range=0.1
        )
        val_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
        )
    elif model_type == 'mobilenetV2':
        train_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        )
        val_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
        )
    elif model_type == 'mobilenetV3Small' or model_type == 'mobilenetV3Large':
        train_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
        )
        val_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input
        )

    # EfficientNet
    elif model_type == 'efficientnetB0' or model_type == 'efficientnetB1' or \
        model_type == 'efficientnetB2' or model_type == 'efficientnetB3' or \
        model_type == 'efficientnetB4' or model_type == 'efficientnetB5' or \
        model_type == 'efficientnetB6' or model_type == 'efficientnetB7':
        train_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        )
        val_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
        )

    # Xception
    elif model_type == 'xception':
        train_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.xception.preprocess_input,
        )
        val_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.xception.preprocess_input
        )

    # EfficientNetV2
    elif model_type == 'efficientnetV2B0' or model_type == 'efficientnetV2B1' or\
        model_type == 'efficientnetV2B2' or model_type == 'efficientnetV2B3' or\
        model_type == 'efficientnetV2S' or model_type == 'efficientnetV2M' or \
        model_type == 'efficientnetV2L':
        train_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input,
        )
        val_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input
        )

    # ResNet - ResNetV2 (50, 101, 152)
    elif model_type == 'resnet50' or model_type == 'resnet101' or model_type == 'resnet152':
        train_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet.preprocess_input,
        )
        val_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet.preprocess_input
        )
    elif model_type == 'resnet50V2' or model_type == 'resnet101V2' or model_type == 'resnet152V2':
        train_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input,
        )
        val_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
        )

    # InceptionV3 and InceptionResNetV2
    elif model_type == 'inceptionV3':
        train_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.inception_v3.preprocess_input,
        )
        val_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.inception_v3.preprocess_input
        )
    elif model_type == 'inceptionresnetV2':
        train_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input,
        )
        val_preprocessor = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input
        )


    train_generators = train_preprocessor.flow(
        x_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    val_generators = val_preprocessor.flow(
        x_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )
    return train_generators, val_generators
