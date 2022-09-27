from statistics import mode
from utils import data_to_list, create_generators
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from Models import custom_model, pre_trainied_model
import matplotlib.pyplot as plt
import os
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-s", "--img_size", type=int, required=False,
                help="Size of Image used to train the model")
ap.add_argument("-b", "--batch_size", type=int, default=32,
                help="batch size of model training")
ap.add_argument("-e", "--epochs", type=int, default=50,
                help="epochs of model training")
ap.add_argument("--model", type=str,  default='mobilenetV2',
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
                help="select model type custom or mobilenetV2,..etc")
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

    # If selected Model is VGGG
    if model_type == 'vgg16' or model_type == 'vgg19':
        img_size = 224

    # If selected Model is MobileNet
    elif model_type == 'mobilenet' or model_type == 'mobilenetV2' or \
        model_type == 'mobilenetV3Small' or model_type == 'mobilenetV3Large':
        img_size = 224

    # If selected Model is EfficientNet B0 - B7
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

    # img_size for EfficientNetV2 B0 to B3 and S, M, L
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
        img_size = 224


    print(f'[INFO] {model_type} Model Expected input size {img_size, img_size, 3}')
    print(f'[INFO] So Taking Input Size as {img_size, img_size, 3}')

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
        x_val, y_val,
        model_type
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
    
    # Pre-trainied Model
    else:
        model = pre_trainied_model(num_class, model_type)

    # Model Summary
    print(f'[INFO] {model_type} Model Summary:')
    print(model.summary())
    
    # Model Training
    print(f'[INFO] {model_type} Model Training Started...')
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
    print('[INFO] Model Evaluation Started...')

    # Evaluate the trained model.
    model_eval_history = model.evaluate(val_generators)

    # Get the loss and accuracy from model_eval_history.
    model_eval_loss, model_eval_accuracy = model_eval_history
    print('Model Evaluation Loss: ', model_eval_loss)
    print('Model Evaluation Accuracy: ', model_eval_accuracy)

    # Define a useful name for our model to make it easy for us while navigating through multiple saved models.
    model_file_name = f'{model_type}_model_loss_{model_eval_loss:.3}_acc_{model_eval_accuracy:.3}.h5'
    model_path = os.path.split(model_path)[0]
    model_full_path = os.path.join(model_path, model_file_name)
    
    # Saved Model
    model.save(model_full_path)
    print(f'[INFO] Successfully Saved model in {model_full_path}')

    # Plot History
    metric_loss = history.history['loss']
    metric_val_loss = history.history['val_loss']
    metric_accuracy = history.history['accuracy']
    metric_val_accuracy = history.history['val_accuracy']

    # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
    epochs = range(len(metric_loss))

    # Plot the Graph.
    plt.plot(epochs, metric_loss, 'blue', label=metric_loss)
    plt.plot(epochs, metric_val_loss, 'red', label=metric_val_loss)
    plt.plot(epochs, metric_accuracy, 'magenta', label=metric_accuracy)
    plt.plot(epochs, metric_val_accuracy, 'green', label=metric_val_accuracy)

    # Y-Axis Limit
    plt.ylim(0, 1.2)

    # Add title to the plot.
    plt.title(str('Model Metrics'))

    # Add legend to the plot.
    plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'])

    # Save Model Metrics Plot
    metrics_name = f'{model_type}_model_metrics.png'
    path_save_metrics = os.path.join(model_path, metrics_name)
    plt.savefig(path_save_metrics, bbox_inches='tight')
    print(f'[INFO] Metrics saved as {path_save_metrics}')

else:
    print(f'[INFO] {model_path} is already Exist')
