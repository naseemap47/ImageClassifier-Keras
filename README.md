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
# Model Training
## You have 2 Options:
- Train on your own **Custom Model**
- Train on **Pre-Trained Model**

### 1. Train on your own Custom Model

Open **Models.py**: <br>
Edit **custom_model** function to your own **Custom Model** Function.

`-i`, `--dataset`: Path to dataset/dir <br>
`-s`, `--img_size`: Size of Image used to train the model <br>
`-b`, `--batch_size`: Batch Size of Model Training <br>
`-e`, `--epochs`: Epochs of Model Training <br>
`--model`: Select Model type custom or mobilenetV2,..etc <br>
- Custom Model: `custom`

`--model_save`: Path to save model.h5

**Example:**
```
python3 train.py --dataset Data/ --img_size 32 --batch_size 16 \
                 --epochs 50 --model custom --model_save model.h5
```

### 2. Train on Pre-Trained Model

#### 🗒️ Note:<br>

I added a **Dense Layer** on last Layer in the Pre-Trained Model, with size **1024**. You can edit that layer, If necessary.<br>
**To Edit**:<br>
Open **Models.py**, go to **Line-192** of **pre_trainied_model** function, there you can find the Dense Layer.

`-i`, `--dataset`: Path to dataset/dir <br>
`-b`, `--batch_size`: Batch Size of Model Training <br>
`-e`, `--epochs`: Epochs of Model Training <br>
`--model`: Select Model type custom or mobilenetV2,..etc <br>

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

`--model_save`: Path to save model.h5

**Image Size** set automatically with respect to selected Pre-Trained Model

**Example:**
```
python3 train.py --dataset Data/ --batch_size 8 --epochs 80 \
                 --model efficientnetB1 --model_save model.h5
```

# Inference

You can Inference on Test **Image, Video and Web-cam** using your saved **Custom Model** or **Pre-Trained Model**. **from a Single Line**. <br>
If you need to Save the output Image or Video. Just need to put `--save` argument in the Last.

**🗒️ Note:**<br>
Please make sure that, You Trained and Inferencing in the same Python Version.<br>
If **NOT**, there is a chance to get **Error: "code = marshal.loads(raw_code) ValueError: bad marshal data (unknown type code)"**.<br>
I got an **Error** when loading **"InceptionResNetV2"** trained model. Becouse I trained on Colab (**python 3.7.14**) and Inference in (**python 3.9**).<br>
The rest of the model didn't get any error. Only I got in **"InceptionResNetV2"** Model.<br>

**Explanation:**<br>
This typically happens when you save a model in one **Python** version (e.g., **3.6**) and then try to load that model in another **Python** version (e.g., **3.9**), as the binary serialization that **Keras** uses ([marshal](https://docs.python.org/3/library/marshal.html)) is not upwards/downwards compatible. Try to install an old version of Python with an appropriate version of the Tensorflow / Keras libraries. If the model was not trained by yourself, you may ask the creators to export the trained models in a different format that doesn't have these problems, like [ONNX](https://onnx.ai/).<br>

**So, its better to train and Inference in the same Python Version.**

## You have 2 Options:
- Inference on your own **Custom Model**
- Inference on **Pre-Trained Model**

### 1. Inference on your own Custom Model

`--img_size`: Size of Image used to Train the model <br>
`-m`, `--model`: Path to saved .h5 model, eg: dir/model.h5 <br>
`--model_type`: Select model type custom or mobilenetV2,vgg16..etc <br>
`-c`, `--conf`: Min prediction confidence to detect pose class (**0<conf<1**) <br>
`--source`: Path to sample image, video or Webcam <br>
`--save`: If need to Save image, video or Webcam <br>

**Example:**
- Image
```
python3 inference.py --img_size 32 --model model.h5 --model_type custom \
                     --conf 0.8 --source test/image.jpg

# to Save output
python3 inference.py --img_size 32 --model model.h5 --model_type custom \
                     --conf 0.8 --source test/image.jpg --save
```
- Video
```
python3 inference.py --img_size 32 --model model.h5 --model_type custom \
                     --conf 0.8 --source test/video.mp4

# to Save output
python3 inference.py --img_size 32 --model model.h5 --model_type custom \
                     --conf 0.8 --source test/video.mp4 --save
```
- Web-cam
```
python3 inference.py --img_size 32 --model model.h5 --model_type custom \
                     --conf 0.8 --source 0

# to Save output
python3 inference.py --img_size 32 --model model.h5 --model_type custom \
                     --conf 0.8 --source 0 --save
```

### 2. Inference on Pre-Trained Model

`-m`, `--model`: Path to saved .h5 model, eg: dir/model.h5 <br>
`--model_type`: Select model type custom or mobilenetV2,vgg16..etc <br>

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

`-c`, `--conf`: Min prediction confidence to detect pose class (**0<conf<1**) <br>
`--source`: Path to sample image, video or Webcam <br>
`--save`: If need to Save image, video or Webcam <br>

**Image Size** set automatically with respect to selected Pre-Trained Model

**Example:**
- Image
```
python3 inference.py --model model.h5 --model_type mobilenet \
                     --conf 0.8 --source test/image.jpg

# to Save output
python3 inference.py --model model.h5 --model_type mobilenet \
                     --conf 0.8 --source test/image.jpg --save
```
- Video
```
python3 inference.py --model model.h5 --model_type mobilenet \
                     --conf 0.8 --source test/video.mp4

# to Save output
python3 inference.py --model model.h5 --model_type mobilenet \
                     --conf 0.8 --source test/video.mp4 --save
```
- Web-cam
```
python3 inference.py --model model.h5 --model_type mobilenet \
                     --conf 0.8 --source 0

# to Save output
python3 inference.py --model model.h5 --model_type mobilenet \
                     --conf 0.8 --source 0 --save
```
