from utils import data_to_list
from sklearn.model_selection import train_test_split


# All image data into a single list
img_list, class_list = data_to_list('Data/')

# Split Data
x_train, x_val, y_train, y_val = train_test_split(img_list, class_list, test_size=0.2)

