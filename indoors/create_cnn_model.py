"""
This script creates a model (map) of the environment from the train dataset using VGG16 CNN
"""

import glob
from natsort import natsorted
import pandas as pd
from torchvision.io import read_image
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

import torch
from PIL import Image
from torchvision import transforms



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
    # weights = AlexNet_Weights.DEFAULT
    # model = alexnet(weights=weights)
    # model.eval()
    # # extract convolutional layers 4+5 (named internally as features.8+10)
    # model = create_feature_extractor(model, {'features.8': 'conv4', 'features.10': 'conv5'})    
    # preprocess = weights.transforms()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=torch.load('dummy.pth').to(device)
    model.eval()

    model = create_feature_extractor(model, {'relu3': 'lenet_descr'})

    #get all train images and coordinates 
    imagesTrain=[]
    imagesTrainCoord=[]
    for file in natsorted(glob.glob('Friburgo/Friburgo_Train/*.jpeg')):
        #img = read_image(file)
        image = Image.open(file).convert('L') #convert to grayscale
        transform = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        #img = img.expand(3,*img.shape[1:])  #turn grayscale image into 3-channel for alexnet input
        input_tensor = transform(image).unsqueeze(0)
        imagesTrain.append(input_tensor)
        x,y=get_xy(file)
        imagesTrainCoord.append([x,y])


    fdTrain1=[]
    fdTrain2=[]
    
    for i in range(len(imagesTrain)):
        print(i)
        

        #get descriptors from CNN   
        #img_path = 'Friburgo/Friburgo_Train/t1152902729.509119_x0.343810_y-0.001590_a-0.006566.jpeg'

        feat=model(imagesTrain[i])
        # c4=feat['conv4']
        # fd1=c4.flatten().detach().numpy()
        # fdTrain1.append(fd1)

        # c5=feat['conv5']
        # fd2=c5.flatten().detach().numpy()
        # fdTrain2.append(fd2)

        out=feat['lenet_descr']
        fd1=out.flatten().detach().numpy()
        fdTrain1.append(fd1)

    #save data in csv files
    # pd.DataFrame(fdTrain1).to_csv("Alexnet_c4_model.csv", index=None)
    # pd.DataFrame(fdTrain2).to_csv("Alexnet_c5_model.csv", index=None)
    pd.DataFrame(fdTrain1).to_csv("lenet_model.csv", index=None)
    

if __name__ == "__main__":
    main()