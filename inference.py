import os
import cv2
from keras.models import load_model
import pandas as pd
import tensorflow as tf
import numpy as np


# CAMERA RESOLUTION
# frameWidth = 640
# frameHeight = 480

# PROBABILITY THRESHOLD
threshold = 0.75
img_size = 224

cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)

# Model
model = load_model('model.h5')
# labels = pd.read_csv('labels.csv')
# traffic_labels = labels['Name']

class_names = sorted(os.listdir('Dataset'))

while True:
    success, img_og = cap.read()
    # img_rgb = cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_og, (img_size, img_size))
    img = img.astype('float32') / 255
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)[0]
    predict = class_names[prediction.argmax()]
    # print(predict)
    prob_value = np.amax(prediction)
    if prob_value > threshold:
        cv2.putText(img_og, "CLASS: ", (20, 35),
                    cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img_og, "PROBABILITY: ", (20, 75),
                    cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img_og, str(predict), (120, 35),
                    cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_og, str(round(prob_value * 100, 2)) + "%",
                    (180, 75), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Webcam', img_og)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
