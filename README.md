# ImageClassifier-Keras
Custom Image Classifier using **Custom model** and **pre-trained models** (like **MobileNet**, **VGG**, **ResNet**, **EfficientNet**,..etc) with help of **Tensorflow** and **Keras**.<br>
Its a single tool all you needed for your **Image Classification** problems.<br>
### Benefits:
- Option to added your own **Custom Model**
- Options to added **All Pre-Trained Models** avalable in Keras Applications

**You can Train or Inference your Data in your Custom Model or in any Pre-Trained Model, "in a single line"**.

# So, Let's Get Started...
## Clone this Repository
```
git clone https://github.com/naseemap47/ImageClassifier-Keras.git
```
### Install Depencies
```
cd ImageClassifier-Keras
pip3 install -r requirements.txt
```
## Collect Data
Inside Data Directory.
The name of Class Directory should be in index number (starting from zero).
Inside each class having its corresponding Image Data.<br>
### Edit **classes.txt** file
Put your Class names in the order corresponding to index number names given to Class Directories.<br>
Example:
**classes.txt**
```
cat
dog
daisy
dandelion
roses
sunflowers
tulips
.....
```
Example:
**Dataset Structure:**
```
├── Dataset
│   ├── 0
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
│   ├── 1
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
.   .
.   .
```
## Model Training
You have 2 Options:
- Train on your own **Custom Model**
- Train on **Pre-Trained Model**

### 1. Train on your own Custom Model
`-i`, `--dataset`: path to dataset/dir <br>
`-s`, `--img_size`: Size of Image used to train the model <br>
`-b`, `--batch_size`: batch size of model training <br>
`-e`, `--epochs`: epochs of model training <br>
`--model`: select model type custom or mobilenetV2,..etc <br>
- Custom Model: `custom`

`--model_save`: path to save model.h5

**Example:**
```
python3 train.py --dataset Data/ --img_size 32 --batch_size 16 --epochs 50 --model custom --model_save model.h5
```

### 2. Train on Pre-Trained Model
`-i`, `--dataset`: path to dataset/dir <br>
`-b`, `--batch_size`: batch size of model training <br>
`-e`, `--epochs`: epochs of model training <br>
`--model`: select model type custom or mobilenetV2,..etc <br>
- Xception: `xception` <br>
- VGG16: `vgg16` <br>
- VGG19: `vgg19` <br>
- ResNet50: `resnet50` <br>
- ResNet101: `resnet101` <br>
- ResNet152: `resnet152` <br>
- ResNet50V2: `resnet50V2` <br>
- ResNet101V2: `resnet101V2` <br>
- ResNet152V2: `resnet152V2` <br>
- InceptionV3: `inceptionV3` <br>
- InceptionResNetV2: `inceptionresnetV2` <br>
- MobileNet: `mobilenet` <br>
- MobileNetV2: `mobilenetV2` <br>
- MobileNetV3Small: `mobilenetV3Small` <br>
- MobileNetV3Large: `mobilenetV3Large` <br>
- DenseNet121: `densenet121` <br>
- DenseNet169: `densenet169` <br>
- DenseNet201: `densenet201` <br>
- NASNetLarge: `nasnetLarge` <br>
- NASNetMobile: `nasnetMobile` <br>
- EfficientNetB0: `efficientnetB0` <br>
- EfficientNetB1: `efficientnetB1` <br>
- EfficientNetB2: `efficientnetB2` <br>
- EfficientNetB3: `efficientnetB3` <br>
- EfficientNetB4: `efficientnetB4` <br>
- EfficientNetB5: `efficientnetB5` <br>
- EfficientNetB6: `efficientnetB6` <br>
- EfficientNetB7: `efficientnetB7` <br>
- EfficientNetV2B0: `efficientnetV2B0` <br>
- EfficientNetV2B1: `efficientnetV2B1` <br>
- EfficientNetV2B2: `efficientnetV2B2` <br>
- EfficientNetV2B3: `efficientnetV2B3` <br>
- EfficientNetV2S: `efficientnetV2S` <br>
- EfficientNetV2M: `efficientnetV2M` <br>
- EfficientNetV2L: `efficientnetV2L` <br>

`--model_save`: path to save model.h5

**Example:**
```
python3 train.py --dataset Data/ --batch_size 8 --epochs 80 --model efficientnetB1 --model_save model.h5
```
