"""
This script takes the hierarchical model
and uses the test dataset to solve the localization problem
"""

import numpy as np
import glob
from natsort import natsorted
import pandas as pd 
from statistics import mean
from math import sqrt
import time
from matplotlib import pyplot as plt
from torchvision.io import read_image
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor

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
        img = read_image(file)
        #img = img.expand(3,*img.shape[1:])  #turn grayscale image into 3-channel for alexnet input
        imagesTest.append(img)
        x,y=get_xy(file)
        imagesTestCoord.append([x,y])

    #get model data from csv 
    mapCoords = pd.read_csv("map_coordinates.csv", header=0).to_numpy()
    fdTrain=pd.read_csv("vgg16_hierarchical_model.csv", header=0).to_numpy()
    fdRepr=pd.read_csv("vgg16_representative_descriptors.csv", header=0).to_numpy()

    #load CNN model
    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights)
    model.eval()
    # extract convolutional layers 4+5 (named internally as features.8+10)
    model = create_feature_extractor(model, {'features.30': 'vgg16_descr'})
    preprocess = weights.transforms()

    distances=[]
    times=[]
    neighbour=[]

    #check all test images
    #for i in range(50):
    for i in range(len(imagesTest)):
        print(i)
        #get starting time each iteration
        startTime=time.time()

        #get descriptor from CNN           
        batch = preprocess(imagesTest[i]).unsqueeze(0)
        feat=model(batch)

        out=feat['vgg16_descr']
        fd=out.flatten().detach().numpy()



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
    pd.DataFrame(zip(distances,times,neighbour), columns=['distance', 'cpu time', 'neighbour'],).to_csv("hierarchical_location_vgg16.csv", index=None)


if __name__ == "__main__":
    main()