# set the numpy seed for better reproducibility
import numpy as np
np.random.seed(42)
import time
import torch
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from skimage.feature import hog
import cv2 as cv
import math

from bayes_filter_loop import get_mov, movement_model, get_new_bel, get_new_bel_correction

def load_cnn_feat_ext():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=torch.load('alexnet_e100.pth').to(device)
    model.eval()

    model = create_feature_extractor(model, {'classifier.2': 'lenet_descr'})

    return model

def movement_model2(map,starting_index,u):

    #prob=np.zeros(len(map))
    offset=20 #how many values around the expected one, we'll consider

    # Calculate the upper and lower indices
    expected_index = (starting_index + u) % len(map)

    # Create the probability density function
    prob = np.zeros(len(map))
    for i in range(len(map)):
        distance = min(abs(i - expected_index), len(map) - abs(i - expected_index))
        if distance <= offset:
            prob[i] = np.exp(-distance)

    # Normalize the probability density function
    prob /= np.sum(prob)

    return prob

def get_new_bel2(map,bel,u):
    new_bel=np.zeros(len(map))

    # For a given movement "u", get the probability of landing in each of the map's positions
    for i in range(len(map)):
        #prob=movement_model(map,starting_pos=map[i,:],u=u)
        prob=movement_model2(map,starting_index=i,u=u)

        new_bel[i]=math.fsum(bel*prob)

    # Correct belief vector so its sum of elements amount to 1 (add difference to highest probability)
    diff=1-math.fsum(new_bel)
    index=np.argmax(new_bel)
    new_bel[index]+=diff

    return new_bel

def get_new_bel3(map,starting_index,N,c):



    # Calculate the upper and lower indices
    length = len(map)

    indices = []
    pdf = np.zeros(length)  # Initialize the pdf array with zeros

    for i in range(-N, N + 1):
        index = (starting_index + i) % length  # Handle wrapping of indices
        distance = min(abs(i), length - abs(i))

        if distance <= c:
            probability = 2  # Assign p1 probability to indices within c distance from the starting index
        else:
            probability = 1  # Assign p2 probability to indices between c and N distance from the starting index

        indices.append(index)
        pdf[index] = probability

    # Normalize the pdf to sum up to 1
    pdf /= np.sum(pdf)

    # print("Indices:")
    # print(indices)

    # print("Probability Density Function:")
    # print(pdf)

    return pdf, indices

def get_new_bel4(model, test_vector, indices):
    pdf_distances = np.zeros(len(model))  # Initialize the pdf_distances array with zeros

    for index in indices:
        model_vector = model[index]
        distance = np.linalg.norm(np.array(test_vector) - np.array(model_vector))  # Calculate Euclidean distance
        pdf_distances[index] = 1 / (distance + 1)  # Define the probability based on distance

    pdf_distances /= np.sum(pdf_distances)

    # print("PDF Distances:")
    # print(pdf_distances)

    return pdf_distances

def main():
    # Initialize external data for loaclization algorithm
    map_coords=pd.read_csv('map_coordinates.csv', header=0).to_numpy()
    test_coords=pd.read_csv('test_coordinates.csv', header=0).to_numpy()
    #hog_offline_results=pd.read_csv('Qevent_hog_descriptor_distances.csv', header=0).to_numpy()
    map_labels=pd.read_csv('labels_gist50.csv', header=0).to_numpy()
    #map_vectors=pd.read_csv('alexnet_e100_model.csv', header=0).to_numpy()
    map_vectors=pd.read_csv('HOG_model.csv', header=0).to_numpy()

    # Get initial belief
    bel=np.full(len(map_coords),fill_value=1/len(map_coords))
    bel=movement_model(map=map_coords,starting_pos=test_coords[0,:],u=[0,0])

    # Initilize results array and save initial belief
    result=np.zeros((len(test_coords),len(map_coords)))
    result[0,:]=bel[:]
    times=[]
    times_full=[]


    test_csv = 'friburgo_test_annotations_file.csv'
    test_df = pd.read_csv(test_csv)

    # Create empty lists to store the predicted labels and ground truth labels
    #predicted_labels = []
    #true_labels = []

    #Load the feature extractor
    #feature_extractor=load_cnn_feat_ext()

    # Adjust window size
    d1=1    # Inner window
    d2=3    # Outer window

    # Loop through the test dataset and combine bayes filter with CNN predictions
    for i in range(1,len(test_df)):
        print(i)
        print('Iteration test image number {number}'.format(number=i))

        # Get starting time each iteration
        start_time_full=time.time()

        # Prediction step
        print('PREDICTION (movement model)')

        likeliest_index=np.argmax(bel)
        print('Starting position: {idx}'.format(idx=likeliest_index))

        #t=1 #number of timestamps between measures, in theory the robot could move 1 index per timestamp
        #bel=get_new_bel2(map_coords,bel,u=t) 
        bel,indices = get_new_bel3(bel, likeliest_index, d2, d1)
        #input('wait')

        print("Indices:")
        print(indices)

        # Correction step
        # Get starting time in each correction step
        start_time=time.time()

        print('CORRECTION (observation model)')
        # Get holistic HOG descriptor
        img_path = test_df.iloc[i, 0]
        image=cv.imread(img_path)
        test_vector, _ = hog(image, orientations=8, pixels_per_cell=(64, 64),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)



        # Update belief
        bel = get_new_bel4(map_vectors,test_vector,indices)       
        # print(bel)
        # input('wait')

        # Get end time after making all computations
        end_time=time.time()

        # Save result in array and later export to csv
        result[i,:]=bel[:]
        times.append(end_time-start_time)
        times_full.append(end_time-start_time_full)
        #confidences.append(confidence)


        # # Take the same test image and get data from hog offline test
        # #prob=hog_offline_results[i+1,:] # i+1 -> next position
        # prob=np.reciprocal(prob)
        # prob=np.nan_to_num(prob, posinf=0)

        # #normalize probability vector
        # prob=prob/math.fsum(prob)
        # bel=get_new_bel_correction(bel,prob)


        # # Save result in array and later export to csv
        # result[i+1,:]=bel[:]

    result[-1] = result[-2]
    # Compute the confusion matrix
    #conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Create a pandas DataFrame from the matrix and export the DataFrame to a CSV file
    #df = pd.DataFrame(conf_matrix)
    #df.to_csv('alexnet_e100_conf_matrix.csv', index=False, header=False)

    pd.DataFrame(result).to_csv('metodo4.csv', index=None)

    #extra_info=np.column_stack((times,confidences))
    #pd.DataFrame(extra_info, columns=['cpu time', 'confidence'],).to_csv('friburgo_hierarchy_extra_info_V2.csv', index=None)
    extra_info=np.column_stack((times,times_full))
    pd.DataFrame(extra_info, columns=['cpu time', 'cpu time (full)'],).to_csv('metodo4_extrainfo.csv', index=None)


if __name__ == "__main__":
    main()
