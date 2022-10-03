# ImageClassifier-Keras
Custom Image Classifier using **Custom model** and **pre-trained models** (like **MobileNet**, **VGG**, **ResNet**, **EfficientNet**,..etc) with help of **Tensorflow** and **Keras**.<br>
Its a single tool all you needed for your **Image Classification** problems.<br>
You can **Inference** on Test **Image, Video and Web-cam** using your saved **Custom Model** or **Pre-Trained Model**. **from a Single Line**. <br>
If you need to **Save** the output **Image or Video**. Just need to put `--save` argument in the Last.<br>

Also you can Train Model using Data from **S3 Bucket** - **"Without Downloading"** <br>
Its same like **Custom** and **Pre-Trained** model Training. <br>
Only few small changes. You need to give extra 3 Arguments. That's all..<br>

Added **MLFlow** to Model training. So that we can Manage the **ML lifecycle**,<br>
**Including**:
- Experimentation
- Reproducibility
- Deployment
- Central Model Registry.

### Features:
- Option to added your own **Custom Model**
- Options to added **All Pre-Trained Models** avalable in Keras Applications
- You can Train your Model using Data from **S3 Bucket** - **"Without Downloading"**
- **MLflow**: Managing the end-to-end **Machine Learning Lifecycle**.

**You can Train or Inference your Data in your Custom Model or in any Pre-Trained Model, "in a single line"**.
## Keras Applications

|Model          |Model Size (MB)|Top-1 Accuracy|Top-5 Accuracy|Parameters|Depth|Time (ms) per inference step (CPU)|Time (ms) per inference step (GPU)|
| ------------- |:-------------:| ------------:| ------------:| --------:| ---:| --------------------------------:|---------------------------------:|
| Xception      | 88            | 79.0%        |94.5%         |22.9M     |81   |109                               |8.1                               |
| VGG16         | 528           |   73.1%      |90.1%         |138.4M    |16   |69.5                              |4.2                               |
|VGG19	        |549            |	71.3%        |	90.0%       |	143.7M   |	19 |	84.8                            |	4.4                              |
|ResNet50	|98	|74.9%	|92.1%	|25.6M	|107	|58.2|	4.6|
|ResNet50V2|	98	|76.0%	|93.0%|	25.6M|	103|	45.6|	4.4|
|ResNet101	|171|	76.4%	|92.8%	|44.7M	|209|	89.6|	5.2|
|ResNet101V2|	171	|77.2%	|93.8%	|44.7M	|205	|72.7	|5.4|
|ResNet152	|232	|76.6%	|93.1%	|60.4M|	311|	127.4|	6.5|
|ResNet152V2|	232	|78.0%	|94.2%	|60.4M	|307	|107.5	|6.6|
|InceptionV3	|92	|77.9%	|93.7%	|23.9M	|189	|42.2|	6.9|
|InceptionResNetV2|	215	|80.3%	|95.3%	|55.9M|	449	|130.2	|10.0|
|MobileNet	|16	|70.4%	|89.5%	|4.3M|	55|	22.6	|3.4|
|MobileNetV2|	14	|71.3%	|90.1%|	3.5M	|105|	25.9	|3.8|
|DenseNet121	|33	|75.0%	|92.3%	|8.1M|	242|	77.1|	5.4|
|DenseNet169|	57	|76.2%	|93.2%	|14.3M	|338	|96.4	|6.3|
|DenseNet201	|80	|77.3%|	93.6%|	20.2M|	402|	127.2|	6.7|
|NASNetMobile	|23	|74.4%	|91.9%|	5.3M	|389	|27.0	|6.7|
|NASNetLarge	|343|	82.5%|	96.0%|	88.9M	|533|	344.5|	20.0|
|EfficientNetB0|	29	|77.1%	|93.3%	|5.3M	|132	|46.0	|4.9|
|EfficientNetB1	|31	|79.1%|	94.4%|	7.9M|	186|	60.2|	5.6|
|EfficientNetB2	|36|	80.1%|	94.9%	|9.2M	|186	|80.8	|6.5|
|EfficientNetB3	|48	|81.6%|	95.7%|	12.3M|	210|	140.0|	8.8|
|EfficientNetB4	|75	|82.9%	|96.4%|	19.5M	|258	|308.3	|15.1|
|EfficientNetB5	|118|	83.6%|	96.7%	|30.6M|	312|	579.2|	25.3|
|EfficientNetB6	|166	|84.0%	|96.8%	|43.3M	|360	|958.1	|40.4|
|EfficientNetB7	|256	|84.3%	|97.0%|	66.7M	|438|	1578.9	|61.6|
|EfficientNetV2B0|	29	|78.7%	|94.3%	|7.2M	|-	|-	|-|
|EfficientNetV2B1	|34|	79.8%|	95.0%|	8.2M|	-|	-|	-|
|EfficientNetV2B2	|42	|80.5%|	95.1%	|10.2M	|-	|-	|-|
|EfficientNetV2B3	|59	|82.0%	|95.8%	|14.5M|	-|	-|	-|
|EfficientNetV2S	|88	|83.9%	|96.7%	|21.6M|	-	|-	|-|
|EfficientNetV2M	|220	|85.3%|	97.4%	|54.4M|	-	|-	|-|
|EfficientNetV2L	|479	|85.7%	|97.5%	|119.0M	|-	|-	|-|

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
#### You have 3 Options:
- Train on your own **Custom Model**
- Train on **Pre-Trained Model**
- Train Model using Data from **S3 Bucket** - **"Without Downloading"**

## 1. Train on your own Custom Model

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

### Train Model using Data from S3 Bucket - "Without Downloading"
Its same like **Custom** model Training.
Only few small changes. You need to give extra 3 Arguments. That's it..

`-n`, `--bucket`: S3 Bucket Name <br>
`-i`, `--dataset`: path to dataset/dir in S3 Bucket <br>
`-s`, `--img_size`: Size of Image used to train the model <br>
`-b`, `--batch_size`: Batch Size of Model Training <br>
`-e`, `--epochs`: Epochs of Model Training <br>
`--model`: Select Model type custom or mobilenetV2,..etc <br>
- Custom Model: `custom`

`--model_save`: Path to save model.h5 <br>
`--aws_region`: AWS Region Name <br>
`--aws_access_key_id`: AWS Access Key ID <br>
`--aws_sec_access_key`: AWS Secret Access Key <br>

**Example:**
```
python3 trainS3.py --bucket my_bucket --dataset ImageData/Train --img_size 320 \
                   --batch_size 8 --epochs 100 --model custom \
                   --model_save model.h5 --aws_region 'ap-south-1' \
                   --aws_access_key_id 'AKDU74NH7MNO2NFEY5NK' \
                   --aws_sec_access_key 'JtyTgFFr/34huiHygUGu&hg7UIHisdhsoit7dsHF'
```

## 2. Train on Pre-Trained Model

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
### Train Model using Data from S3 Bucket - "Without Downloading"

Its same like **Pre-Trained** model Training.
Only few small changes. You need to give extra 3 Arguments. That's it..

`-n`, `--bucket`: S3 Bucket Name <br>
`-i`, `--dataset`: path to dataset/dir in S3 Bucket <br>
`-b`, `--batch_size`: Batch Size of Model Training <br>
`-e`, `--epochs`: Epochs of Model Training <br>
`--model`: Select Model type custom or mobilenetV2,..etc <br>
- All types of pre-trained models given above

`--model_save`: Path to save model.h5 <br>
`--aws_region`: AWS Region Name <br>
`--aws_access_key_id`: AWS Access Key ID <br>
`--aws_sec_access_key`: AWS Secret Access Key <br>

**Example:**
```
python3 trainS3.py --bucket my_bucket --dataset ImageData/Train \
                   --batch_size 8 --epochs 100 --model vgg16 \
                   --model_save model.h5 --aws_region 'ap-south-1' \
                   --aws_access_key_id 'AKDU74NH7MNO2NFEY5NK' \
                   --aws_sec_access_key 'JtyTgFFr/34huiHygUGu&hg7UIHisdhsoit7dsHF'
```
## MLFlow UI
**MLflow** is an open source platform for managing the end-to-end machine learning lifecycle <br>
terminal is in the same directory that contains mlruns, and
type the following:
```
mlflow ui

# OR
mlflow ui -p 1234
```
The command mlflow ui hosts the MLFlow UI locally on the default
port of **5000**.<br>
However, the options `-p 1234` tell it that you want to host it specifically on the port **1234**.<br>

open a browser and type in http://localhost:1234 or
http://127.0.0.1:1234

# Inference

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