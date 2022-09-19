from utils import data_to_list, create_generators
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from Model import Model
import os
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-b", "--batch_size", type=int, default=32,
                help="batch size of model training")
ap.add_argument("-e", "--epochs", type=int, default=50,
                help="epochs of model training")
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to save model.h5")


args = vars(ap.parse_args())
path_to_dir = args["dataset"]
batch_size = args['batch_size']
epochs = args['epochs']
model_path = args['model']


# All image data into a single list
img_list, class_list, num_class = data_to_list(path_to_dir)

# Split Data
x_train, x_val, y_train, y_val = train_test_split(img_list, class_list, test_size=0.2)

# Preprocessing
train_generators, val_generators = create_generators(
                                                    batch_size*2, num_class,
                                                    x_train, y_train,
                                                    x_val, y_val
                                                )

# Callbacks
early_stopping = EarlyStopping(
    min_delta=0.001,
    patience=10,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

# Model
model = Model(no_classes=num_class)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_generators,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=val_generators,
    callbacks=[early_stopping]
)

if os.path.isfile(model_path) is False:
    model.save(model_path)
    print(f'[INFO] Successfully Saved model in {model_path}')

