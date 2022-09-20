from utils import data_to_list, create_generators
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from Models import custom_model, mobilenet_v2_model
import os
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-s", "--img_size", type=int, required=True,
                help="Size of Image used to train the model")
ap.add_argument("-b", "--batch_size", type=int, default=32,
                help="batch size of model training")
ap.add_argument("-e", "--epochs", type=int, default=50,
                help="epochs of model training")
ap.add_argument("--model", type=str,  default='mobilenetV2',
                choices=['custom', 'mobilenetV2'],
                help="select model type custom or mobilenetV2")
ap.add_argument("--model_save", type=str, required=True,
                help="path to save model.h5")

args = vars(ap.parse_args())
path_to_dir = args["dataset"]
img_size = args['img_size']
batch_size = args['batch_size']
epochs = args['epochs']
model_type = args["model"]
model_path = args['model_save']

if os.path.isfile(model_path) is False:

    # If selected Model is Mobilenet V2
    if model_type == 'mobilenetV2':
        img_size = 224

    # All image data into a single list
    print('[INFO] Image Data Extraction Started...')
    img_list, class_list, num_class = data_to_list(path_to_dir, img_size)
    print('[INFO] Image Data Extraction Completed...')

    # Split Data
    x_train, x_val, y_train, y_val = train_test_split(
        img_list, class_list, test_size=0.2)

    # Preprocessing
    print('[INFO] Image Data Preprocessing Started...')
    train_generators, val_generators = create_generators(
        batch_size*2, num_class,
        x_train, y_train,
        x_val, y_val
    )
    print('[INFO] Image Data Preprocessing Completed...')

    # Callbacks
    early_stopping = EarlyStopping(
        min_delta=0.001,
        patience=10,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )

    # Choose Model
    if model_type == 'custom':
        model = custom_model(num_class, img_size)
    elif model_type == 'mobilenetV2':
        model = mobilenet_v2_model(num_class)
    # Model Training
    print('[INFO] Model Training Started...')
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
    print('[INFO] Model Training Completed...')

    # Saved Model
    model.save(model_path)
    print(f'[INFO] Successfully Saved model in {model_path}')

else:
    print(f'[INFO] {model_path} is already Exist')
