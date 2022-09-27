import os
import cv2
from keras.models import load_model
import tensorflow as tf
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("--img_size", type=int, required=False,
                help="Size of Image used to train the model")                
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to saved .h5 model, eg: dir/model.h5")
ap.add_argument("--model_type", type=str,  default='mobilenetV2',
                choices=[
                    'custom', 'vgg16', 'vgg19', 'mobilenet',
                    'mobilenetV2', 'mobilenetV3Small', 'mobilenetV3Large',
                    'efficientnetB0', 'efficientnetB1', 'efficientnetB2',
                    'efficientnetB3', 'efficientnetB4', 'efficientnetB5',
                    'efficientnetB6', 'efficientnetB7', 'xception',
                    'efficientnetV2B0', 'efficientnetV2B1', 'efficientnetV2B2',
                    'efficientnetV2B3', 'efficientnetV2S', 'efficientnetV2M',
                    'efficientnetV2L', 'resnet50', 'resnet101', 'resnet152',
                    'resnet50V2', 'resnet101V2', 'resnet152V2', 'inceptionV3',
                    'inceptionresnetV2', 'densenet121', 'densenet169', 'densenet201'
                ],
                help="select model type custom or mobilenetV2,vgg16..etc")
ap.add_argument("-c", "--conf", type=float, required=True,
                help="min prediction conf to detect pose class (0<conf<1)")
ap.add_argument("--source", type=str, required=True,
                help="path to sample image")
ap.add_argument("--save", action='store_true',
                help="Save video")

args = vars(ap.parse_args())
source = args["source"]
path_saved_model = args["model"]
model_type = args["model_type"]
threshold = args["conf"]
save = args['save']
img_size = args['img_size']

##############################

# If selected Model is VGGG
if model_type == 'vgg16' or model_type == 'vgg19':
    img_size = 224

# If selected Model is MobileNet
elif model_type == 'mobilenet' or model_type == 'mobilenetV2' or \
    model_type == 'mobilenetV3Small' or model_type == 'mobilenetV3Large':
    img_size = 224

# If selected Model is EfficientNet
elif model_type == 'efficientnetB0':
    img_size = 224
elif model_type == 'efficientnetB1':
    img_size = 240
elif model_type == 'efficientnetB2':
    img_size = 260
elif model_type == 'efficientnetB3':
    img_size = 300
elif model_type == 'efficientnetB4':
    img_size = 380
elif model_type == 'efficientnetB5':
    img_size = 456
elif model_type == 'efficientnetB6':
    img_size = 528
elif model_type == 'efficientnetB7':
    img_size = 600

# img_size for Xception Model
elif model_type == 'xception':
    img_size = 299

# EfficientNetV2 B0 to B3 and S, M, L
elif model_type == 'efficientnetV2B0':
    img_size = 224

elif model_type == 'efficientnetV2B1':
    img_size = 240

elif model_type == 'efficientnetV2B2':
    img_size = 260

elif model_type == 'efficientnetV2B3':
    img_size = 300

elif model_type == 'efficientnetV2S':
    img_size = 384

elif model_type == 'efficientnetV2M' or model_type == 'efficientnetV2L':
    img_size = 480

# ResNet - ResNetV2 (50, 101, 152)
elif model_type == 'resnet50' or model_type == 'resnet101' or model_type == 'resnet152' or \
    model_type == 'resnet50V2' or model_type == 'resnet101V2' or model_type == 'resnet152V2':
    img_size = 224

# InceptionV3 and InceptionResNetV2
elif model_type == 'inceptionV3' or model_type == 'inceptionresnetV2':
    img_size = 299

# DenseNet
elif model_type == 'densenet121' or model_type == 'densenet169' or model_type == 'densenet201':
    img_size == 224


print(f'[INFO] {model_type} Model Expected input size {img_size, img_size, 3}')
print(f'[INFO] So Taking Input Size as {img_size, img_size, 3}')

##############################

# Model
saved_model = load_model(path_saved_model)
f = open('classes.txt')
class_names = f.read().splitlines()

###### Image ######
if source.endswith(('.jpg', '.jpeg', '.png')):
    path_to_img = source
    img_og = cv2.imread(path_to_img)
    img_resize = cv2.resize(img_og, (img_size, img_size))
    h, w, _ = img_og.shape
    img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

    # Custom Model
    if model_type == 'custom':
        img = img_rgb.astype('float32') / 255
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
    else:
        img = tf.keras.preprocessing.image.img_to_array(img_rgb)
        img = np.expand_dims(img, axis=0)
        # VGG16
        if model_type == 'vgg16':
            img = tf.keras.applications.vgg16.preprocess_input(img)

        # VGG19
        elif model_type == 'vgg19':
            img = tf.keras.applications.vgg19.preprocess_input(img)

        # MobileNet
        elif model_type == 'mobilenet':
            img = tf.keras.applications.mobilenet.preprocess_input(img)

        # MobileNetV2
        elif model_type == 'mobilenetV2':
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

        # MobileNetV3Small & MobileNetV3Large
        elif model_type == 'mobilenetV3Small' or model_type == 'mobilenetV3Large':
            img = tf.keras.applications.mobilenet_v3.preprocess_input(img)

        # EfficientNet B0 to B7
        elif model_type == 'efficientnetB0' or model_type == 'efficientnetB1' or \
            model_type == 'efficientnetB2' or model_type == 'efficientnetB3' or \
            model_type == 'efficientnetB4' or model_type == 'efficientnetB5' or \
            model_type == 'efficientnetB6' or model_type == 'efficientnetB7':
            img = tf.keras.applications.efficientnet.preprocess_input(img)

        # Xception
        elif model_type == 'xception':
            img = tf.keras.applications.xception.preprocess_input(img)

        # EfficientNetV2 B0 to B3 and S, M, L
        elif model_type == 'efficientnetV2B0' or model_type == 'efficientnetV2B1' or\
            model_type == 'efficientnetV2B2' or model_type == 'efficientnetV2B3' or\
            model_type == 'efficientnetV2S' or model_type == 'efficientnetV2M' or \
            model_type == 'efficientnetV2L':
            img = tf.keras.applications.efficientnet_v2.preprocess_input(img)

        # ResNet (50, 101, 152)
        elif model_type == 'resnet50' or model_type == 'resnet101' or model_type == 'resnet152':
            img = tf.keras.applications.resnet.preprocess_input(img)

        # ResNetV2 (50, 101, 152)
        elif model_type == 'resnet50V2' or model_type == 'resnet101V2' or model_type == 'resnet152V2':
            img = tf.keras.applications.resnet_v2.preprocess_input(img)

        # InceptionV3
        elif model_type == 'inceptionV3':
            img = tf.keras.applications.inception_v3.preprocess_input(img)

        # InceptionResNetV2
        elif model_type == 'inceptionresnetV2':
            img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)

        # DenseNet
        elif model_type == 'densenet121' or model_type == 'densenet169' or model_type == 'densenet201':
            img = tf.keras.applications.densenet.preprocess_input(img)


    prediction = saved_model.predict(img)[0]
    predict = class_names[prediction.argmax()]
    print('[INFO] Predicted Class: ', predict)
    prob_value = np.amax(prediction)

    # if resize is less than original size
    if img_size>h:
        img_og = img_resize
        h, w, _ = img_og.shape

    if prob_value > threshold:

        x_axis = int(w/11.2)
        y1 = int(h/6.4)
        y2 = int(h/3)
        y3 = int(h/4)
        y4 = int(h/2.35)

        font_size = int(((h+w)/2)/224)
        font_thickness = int(font_size*2)
        
        cv2.putText(img_og, "CLASS: ", (x_axis, y1),
                    cv2.FONT_HERSHEY_PLAIN, font_size,
                    (255, 0, 255), font_thickness, cv2.LINE_AA)
        cv2.putText(img_og, "PROBABILITY: ", (x_axis, y2),
                    cv2.FONT_HERSHEY_PLAIN, font_size,
                    (255, 0, 255), font_thickness, cv2.LINE_AA)
        cv2.putText(img_og, str(predict), (x_axis, y3),
                    cv2.FONT_HERSHEY_PLAIN, font_size,
                    (0, 255, 0), font_thickness, cv2.LINE_AA)
        cv2.putText(img_og, str(round(prob_value * 100, 2)) + "%",
                    (x_axis, y4), cv2.FONT_HERSHEY_PLAIN, font_size,
                    (0, 255, 0), font_thickness, cv2.LINE_AA)

    if save:
        os.makedirs('ImageOutput', exist_ok=True)
        img_full_name = os.path.split(path_to_img)[1]
        img_name = os.path.splitext(img_full_name)[0]
        path_to_save_img = f'ImageOutput/{img_name}.jpg'
        cv2.imwrite(f'{path_to_save_img}', img_og)
        print(f'[INFO] Output Image Saved in {path_to_save_img}')

    cv2.imshow('Image', img_og)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    print('[INFO] Inference on Test Image is Ended...')


####### Video or Camera ########
else:
    # Web-cam
    if source.isnumeric():
        source = int(source)

    cap = cv2.VideoCapture(source)
    source_width = int(cap.get(3))
    source_height = int(cap.get(4))

    # Write Video
    if save:
        out_video = cv2.VideoWriter('output.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, (source_width, source_height))

    while True:
        success, img_og = cap.read()
        if not success:
            print('[ERROR] Failed to Read Video feed')
            break
        img_resize = cv2.resize(img_og, (img_size, img_size))
        h, w, _ = img_og.shape
        img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

        # Custom Model
        if model_type == 'custom':
            img = img_rgb.astype('float32') / 255
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
        else:
            img = tf.keras.preprocessing.image.img_to_array(img_rgb)
            img = np.expand_dims(img, axis=0)
            # VGG16
            if model_type == 'vgg16':
                img = tf.keras.applications.vgg16.preprocess_input(img)

            # VGG19
            elif model_type == 'vgg19':
                img = tf.keras.applications.vgg19.preprocess_input(img)

            # MobileNet
            elif model_type == 'mobilenet':
                img = tf.keras.applications.mobilenet.preprocess_input(img)

            # MobileNetV2
            elif model_type == 'mobilenetV2':
                img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

            # MobileNetV3Small & MobileNetV3Large
            elif model_type == 'mobilenetV3Small' or model_type == 'mobilenetV3Large':
                img = tf.keras.applications.mobilenet_v3.preprocess_input(img)

            # EfficientNet B0 to B7
            elif model_type == 'efficientnetB0' or model_type == 'efficientnetB1' or \
                model_type == 'efficientnetB2' or model_type == 'efficientnetB3' or \
                model_type == 'efficientnetB4' or model_type == 'efficientnetB5' or \
                model_type == 'efficientnetB6' or model_type == 'efficientnetB7':
                img = tf.keras.applications.efficientnet.preprocess_input(img)

            # Xception
            elif model_type == 'xception':
                img = tf.keras.applications.xception.preprocess_input(img)

            # EfficientNetV2 B0 to B3 and S, M, L
            elif model_type == 'efficientnetV2B0' or model_type == 'efficientnetV2B1' or\
                model_type == 'efficientnetV2B2' or model_type == 'efficientnetV2B3' or\
                model_type == 'efficientnetV2S' or model_type == 'efficientnetV2M' or \
                model_type == 'efficientnetV2L':
                img = tf.keras.applications.efficientnet_v2.preprocess_input(img)

            # ResNet (50, 101, 152)
            elif model_type == 'resnet50' or model_type == 'resnet101' or model_type == 'resnet152':
                img = tf.keras.applications.resnet.preprocess_input(img)

            # ResNetV2 (50, 101, 152)
            elif model_type == 'resnet50V2' or model_type == 'resnet101V2' or model_type == 'resnet152V2':
                img = tf.keras.applications.resnet_v2.preprocess_input(img)

            # InceptionV3
            elif model_type == 'inceptionV3':
                img = tf.keras.applications.inception_v3.preprocess_input(img)

            # InceptionResNetV2
            elif model_type == 'inceptionresnetV2':
                img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)

            # DenseNet
            elif model_type == 'densenet121' or model_type == 'densenet169' or \
                model_type == 'densenet201':
                img = tf.keras.applications.densenet.preprocess_input(img)


        # Prediction
        prediction = saved_model.predict(img)[0]
        predict = class_names[prediction.argmax()]
        # print(predict)
        prob_value = np.amax(prediction)

        # if resize is less than original size
        if img_size>h:
            img_og = img_resize
            h, w, _ = img_og.shape

        if prob_value > threshold:

            x_axis = int(w/11.2)
            y1 = int(h/6.4)
            y2 = int(h/3)
            y3 = int(h/4)
            y4 = int(h/2.35)

            font_size = int(((h+w)/2)/224)
            font_thickness = int(font_size*2)
            
            cv2.putText(img_og, "CLASS: ", (x_axis, y1),
                        cv2.FONT_HERSHEY_PLAIN, font_size,
                        (255, 0, 255), font_thickness, cv2.LINE_AA)
            cv2.putText(img_og, "PROBABILITY: ", (x_axis, y2),
                        cv2.FONT_HERSHEY_PLAIN, font_size,
                        (255, 0, 255), font_thickness, cv2.LINE_AA)
            cv2.putText(img_og, str(predict), (x_axis, y3),
                        cv2.FONT_HERSHEY_PLAIN, font_size,
                        (0, 255, 0), font_thickness, cv2.LINE_AA)
            cv2.putText(img_og, str(round(prob_value * 100, 2)) + "%",
                        (x_axis, y4), cv2.FONT_HERSHEY_PLAIN, font_size,
                        (0, 255, 0), font_thickness, cv2.LINE_AA)

        # Write Video
        if save:
            video_write_size = cv2.resize(img_og, (source_width, source_height))
            out_video.write(video_write_size)

        cv2.imshow('Video Output', img_og)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if save:
        out_video.release()
        print("[INFO] Out video Saved as 'output.avi'")
    cv2.destroyAllWindows()
    print('[INFO] Inference on Videostream is Ended...')
