import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # GRAPHING AND VISUALIZATIONS
import os
import cv2
from sklearn import preprocessing # for data preprocess, e.g label encoding
from sklearn.model_selection import train_test_split # for splitting train data for validation
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

class Image_dataset:

    def __init__(self):
        self.train_set = list()
        self.train_labels = list()
        self.scale = 70

    # load training sets from image files
    def load_train_file(self,path):
        dirs = os.listdir(path) # read images from given file
        for dir in dirs:
            count = 0
            for img in os.listdir(os.path.join(path,dir)):
                img = os.path.join(path,dir,img)
                self.train_set.append(cv2.resize(cv2.imread(img),(self.scale,self.scale)))
                self.train_labels.append(dir)
                count += 1
            print(f"{dir}: load {count} images done!")
        self.train_set = np.asarray(self.train_set)
        self.train_labels = pd.DataFrame(self.train_labels)
    
    # convert image to hsv, remove background and noise
    def clean_img(self):
        new_train = []
        for i in self.train_set:
            blurr = cv2.GaussianBlur(i,(5,5),0)
            hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)
            # GREEN PARAMETERS
            lower = (25,40,50)
            upper = (75,255,255)
            mask = cv2.inRange(hsv,lower,upper)
            struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
            mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)
            boolean = mask>0
            new = np.zeros_like(i,np.uint8)
            new[boolean] = i[boolean]
            new_train.append(new)
        self.train_set = np.asarray(new_train)
    
    # use sklearn to do label encoding
    def label_encoding(self):
        labels = preprocessing.LabelEncoder()
        labels.fit(self.train_labels[0]) # collect label from the datasets of images
        # tranform label into binary format
        encoded_label = labels.transform(self.train_labels[0])
        binary_label = np_utils.to_categorical(encoded_label)
        self.train_labels = binary_label

        # split the train data to prevent overfitting
    def split_data(self):
        self.train_set = self.train_set/255
        x_train, x_test, y_train, y_test = train_test_split(self.train_set, self.train_labels, test_size=0.1, random_state=self.seed, stratify=self.train_labels)
        # prevent overfitting
        # ImageDataGenerator() randomly changes the characteristics of images and provides randomness in the data
        generator = ImageDataGenerator(rotation_range = 180,zoom_range = 0.1,width_shift_range = 0.1,height_shift_range = 0.1,horizontal_flip = True,vertical_flip = True)
        generator.fit(x_train)

if __name__ == "__main__":
    test = Image_dataset()
    test.load_train_file('train')
    #test.clean_img()
    test.label_encoding()

    
