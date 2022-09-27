# ImageClassifier-Keras
Custom Image Classifier using **Custom model** and **pre-trained models** like **MobileNet**, **VGG**, **ResNet**, **EfficientNet**,..etc with help of Tensorflow and Keras.<br>
Its a single tool all you needed for your **Image Classification**.
Its having option to added your own model and also added all pre-trained models avalable in Keras Applications.<br>
**You can train your Data in your custom model or in any pre-trained model, "in a single line"**.

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
Inside each class having its corresponding Image Data.

Edit **classes.txt** file:<br>
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
