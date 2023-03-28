"""
This script takes the model (map) of the environment created using the HOG descriptor
and compares against every image in the test dataset to get each distance
"""

import numpy as np
import cv2 as cv
from skimage.feature import hog
import glob
from natsort import natsorted
import pandas as pd 


def get_xy(name):
    lista=name.split('_',-1)
    for word in lista:
        if word[0]=='x':
            x=word[1:]
            x=float(x)
        if word[0]=='y':
            y=word[1:]
            y=float(y)
    return x,y   

def main():
    #get all test images
    imagesTest=[]
    imagesTestCoord=[]
    for file in natsorted(glob.glob('Friburgo/Friburgo_Test_ext/*.jpeg')):
        imagesTest.append(cv.imread(file))
        x,y=get_xy(file)
        imagesTestCoord.append([x,y])


    #get model data from csv files
    fdTrain=pd.read_csv('BIS_HOG_model.csv', header=0).to_numpy()


    distances=np.zeros((len(imagesTest),len(fdTrain)))


    #check all test images
    for i in range(len(imagesTest)):
        print(i)

        #get image descriptor
        fd, hog_image = hog(imagesTest[i], orientations=8, pixels_per_cell=(64, 64),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)

        #compare against model
        for j in range(len(fdTrain)):
            dist = np.linalg.norm(fd-fdTrain[j])

            distances[i,j]=dist


    pd.DataFrame(distances).to_csv('BIS_hog_descriptor_distances.csv', index=None)


if __name__ == '__main__':
    main()