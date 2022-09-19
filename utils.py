import cv2
import numpy as np
import os
import glob


def data_to_list(path_to_data):
    images = []
    class_var = []
    class_name_list = os.listdir(path_to_data)
    class_name_list = sorted(class_name_list)

    for class_name in class_name_list:
        img_path_list = glob.glob(class_name + '/*.jpg') + \
                        glob.glob(class_name + '/*.jpeg') + \
                        glob.glob(class_name + '/*.png')
        
        for img_path in img_path_list:
            img = cv2.imread(img_path)
            images.append(img)
            class_var.append(class_name)
    images = np.array(images)
    class_var = np.array(class_var)
    return images, class_var