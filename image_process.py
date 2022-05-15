import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # GRAPHING AND VISUALIZATIONS
import os
import cv2

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
        getEx = True
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
            # show image for one time
            if getEx:
                plt.subplot(2,3,1);plt.imshow(i) # ORIGINAL
                plt.subplot(2,3,2);plt.imshow(blurr) # BLURRED
                plt.subplot(2,3,3);plt.imshow(hsv) # HSV CONVERTED
                plt.subplot(2,3,4);plt.imshow(mask) # MASKED
                plt.subplot(2,3,5);plt.imshow(boolean) # BOOLEAN MASKED
                plt.subplot(2,3,6);plt.imshow(new) # NEW PROCESSED IMAGE
                plt.show()
                getEx = False
        self.train_set = np.asarray(new_train)
        # CLEANED IMAGES
        for i in range(8):
            plt.subplot(2,4,i+1)
            plt.imshow(self.train_set[i])
    
if __name__ == "__main__":
    test = Image_dataset()
    test.load_train_file('train')
    test.clean_img()

    
