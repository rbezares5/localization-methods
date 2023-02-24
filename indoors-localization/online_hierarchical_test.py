"""
This script takes the hierarchical model
and uses the test dataset to solve the localization problem
"""

import numpy as np
import cv2 as cv
from skimage.feature import hog
import glob
from natsort import natsorted
import pandas as pd 
from statistics import mean
from math import sqrt
import time
from matplotlib import pyplot as plt

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


    #get model data from csv 
    mapCoords = pd.read_csv("model_coordinates.csv", header=0).to_numpy()
    fdTrain=pd.read_csv("HOG_hierarchical_model.csv", header=0).to_numpy()
    fdRepr=pd.read_csv("HOG_representative_descriptors.csv", header=0).to_numpy()
    #print(fdRepr)


    distances=[]
    times=[]
    neighbour=[]

    #check all test images
    #for i in range(50):
    for i in range(len(imagesTest)):
        print(i)
        #get starting time each iteration
        startTime=time.time()

        #get image descriptor
        fd, hog_image = hog(imagesTest[i], orientations=8, pixels_per_cell=(64, 64),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)

        #compare against representatives only to get corresponding cluster (rough loaction)
        cluster=-1
        minDist=100
        for j in range(len(fdRepr)):
            dist = np.linalg.norm(fd-fdRepr[j][1:])

            if dist<minDist:
                minDist=dist
                cluster=fdRepr[j][0]

        #print(cluster)

        #compare against model, but only those from the same cluster
        minj=-1
        minDist=100
        for j in range(len(fdTrain)):
            if fdTrain[j][0]==cluster:
                dist = np.linalg.norm(fd-fdTrain[j][1:])

                if dist<minDist:
                    minDist=dist
                    minj=j

        #get chosen image (prediction) and its xy coords from csv
        [xTrain,yTrain]=mapCoords[minj,:]

        #get end time after making prediction
        endTime=time.time()

        #compute error distance
        xTest=imagesTestCoord[i][0]
        yTest=imagesTestCoord[i][1]
        metricDist=sqrt((xTrain-xTest)**2+(yTrain-yTest)**2)

        distances.append(metricDist)
        times.append(endTime-startTime)

        #chosen neighbor histogram list
        #first, make a list of all metric distances
        distList=[]
        for k in range(len(fdTrain)):
            [xMap,yMap]=mapCoords[k,:]
            dist=sqrt((xMap-xTest)**2+(yMap-yTest)**2)
            distList.append(dist)

        #then, determine the order of each distance, where 0 is the closest
        indexList=np.zeros(len(distList))
        for k in reversed(range(len(distList))):
            max_value = max(distList)
            index = distList.index(max_value)
            indexList[index]=k
            distList[index]=0

        #get the index corresponding to the prediction made earlier
        neighbour.append(indexList[minj])


    #print(distances)
    print(mean(distances))
    #print(times)
    print(mean(times))
    #print(neighbour)


    
    # Show plot
    plt.subplot(2,1,1)
    plt.hist(neighbour, bins=[0, 50, 100, 200, 300, 400])

    plt.subplot(2,1,2)
    plt.hist(neighbour, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.show()

    #save result data in csv file
    pd.DataFrame(zip(distances,times,neighbour), columns=['distance', 'cpu time', 'neighbour'],).to_csv("hierarchical_location_hog.csv", index=None)


if __name__ == "__main__":
    main()