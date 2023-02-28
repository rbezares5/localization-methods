"""
This script creates a model (map) of the environment from the train dataset using Alexnet CNN
"""

import glob
from natsort import natsorted
import pandas as pd
from torchvision.io import read_image
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

def get_xy(name):
    lista=name.split('_',4)
    for word in lista:
        if word[0]=='x':
            x=word[1:]
        if word[0]=='y':
            y=word[1:]
    return x,y        


def main():

    #load CNN model
    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    model.eval()
    # extract convolutional layers 4+5 (named internally as features.8+10)
    model = create_feature_extractor(model, {'features.8': 'conv4', 'features.10': 'conv5'})    
    preprocess = weights.transforms()

    #get all train images and coordinates 
    imagesTrain=[]
    imagesTrainCoord=[]
    for file in natsorted(glob.glob('Friburgo/Friburgo_Train/*.jpeg')):
        img = read_image(file)
        img = img.expand(3,*img.shape[1:])  #turn grayscale image into 3-channel for alexnet input
        imagesTrain.append(img)
        x,y=get_xy(file)
        imagesTrainCoord.append([x,y])


    fdTrain1=[]
    fdTrain2=[]
    
    for i in range(len(imagesTrain)):
        print(i)
        

        #get descriptors from CNN   
        batch = preprocess(imagesTrain[i]).unsqueeze(0)
        feat=model(batch)
        c4=feat['conv4']
        fd1=c4.flatten().detach().numpy()
        fdTrain1.append(fd1)

        c5=feat['conv5']
        fd2=c5.flatten().detach().numpy()
        fdTrain2.append(fd2)

    #save data in csv files
    pd.DataFrame(fdTrain1).to_csv("Alexnet_c4_model.csv", index=None)
    pd.DataFrame(fdTrain2).to_csv("Alexnet_c5_model.csv", index=None)
    

if __name__ == "__main__":
    main()