"""
This script creates a model (map) of the environment from the train dataset using the HOG descriptor
"""

import cv2 as cv
from skimage.feature import hog
import glob
from natsort import natsorted
import pandas as pd

def get_xy(name):
    lista=name.split('_',4)
    for word in lista:
        if word[0]=='x':
            x=word[1:]
        if word[0]=='y':
            y=word[1:]
    return x,y        


def main():
    #get all train images and coordinates 
    imagesTrain=[]
    imagesTrainCoord=[]
    for file in natsorted(glob.glob('Friburgo/Friburgo_Train/*.jpeg')):
        imagesTrain.append(cv.imread(file))
        x,y=get_xy(file)
        imagesTrainCoord.append([x,y])

    #get hog descriptor of each image
    ppc=64
    fdTrain=[]
    for i in range(len(imagesTrain)):
        print(i)
        fd, hog_image = hog(imagesTrain[i], orientations=8, pixels_per_cell=(ppc, ppc),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)
        
        fdTrain.append(fd)

    #save data in csv files
    pd.DataFrame(fdTrain).to_csv("HOG_model.csv", index=None)
    pd.DataFrame(imagesTrainCoord, columns=['x', 'y'],).to_csv("model_coordinates.csv", index=None)
    

if __name__ == "__main__":
    main()