import cv2
import numpy as np
import os
import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


def data_to_list(path_to_data, img_size):
    images = []
    class_var = []
    class_name_list = os.listdir(path_to_data)
    class_name_list = sorted(class_name_list)
    num_class = len(class_name_list)

    for class_name in class_name_list:
        img_path_list = glob.glob(class_name + '/*.jpg') + \
                        glob.glob(class_name + '/*.jpeg') + \
                        glob.glob(class_name + '/*.png')
        
        for img_path in img_path_list:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            class_var.append(class_name)
        
        print(f'[INFO] Extracted {class_name}')

    images = np.array(images)
    class_var = np.array(class_var)

    # class names to categorical
    class_var = np.unique(class_var, return_inverse=True)[1]
    class_var = to_categorical(class_var)

    return images, class_var, num_class


def create_generators(batch_size, no_class,
                      x_train, y_train,
                      x_val, y_val):
    # to_categorical
    y_train = to_categorical(y_train, no_class)
    y_val = to_categorical(y_val, no_class)

    # Preprocessor
    train_preprocessor = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    val_preprocessor = ImageDataGenerator(rescale=1 / 255)

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

