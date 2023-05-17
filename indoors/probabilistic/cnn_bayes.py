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
from scipy.spatial.distance import cdist

from bayes_filter_loop import get_mov, movement_model, get_new_bel, get_new_bel_correction

def load_cnn_feat_ext():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=torch.load('alexnet_e100.pth').to(device)
    model.eval()

    model = create_feature_extractor(model, {'classifier.2': 'lenet_descr'})

    return model

def main():
    # Initialize external data for loaclization algorithm
    map_coords=pd.read_csv('map_coordinates.csv', header=0).to_numpy()
    test_coords=pd.read_csv('test_coordinates.csv', header=0).to_numpy()
    #hog_offline_results=pd.read_csv('Qevent_hog_descriptor_distances.csv', header=0).to_numpy()
    map_labels=pd.read_csv('labels_gist50.csv', header=0).to_numpy()
    map_vectors=pd.read_csv('alexnet_e100_model.csv', header=0).to_numpy()

    # Get initial belief
    bel=np.full(len(map_coords),fill_value=1/len(map_coords))
    bel=movement_model(map=map_coords,starting_pos=test_coords[0,:],u=[0,0])

    # Initilize results array and save initial belief
    result=np.zeros((len(test_coords),len(map_coords)))
    result[0,:]=bel[:]
    times=[]
    confidences=[]

    # For new algorithm, firstly we find the rough location, that is, we take 
    # an image and assign it to a cluster decided by a prediction from our CNN

    # Set the device we will be using to test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #print('LOAD MODEL FROM DISK')
    model=torch.load('alexnet_e100.pth').to(device)

    #print('EVALUATE MODEL')
    model.eval()

    #print('MAKING PREDICTION')
    # Load the test dataset
    test_csv = 'friburgo_test_annotations_file.csv'
    test_df = pd.read_csv(test_csv)

    # Create empty lists to store the predicted labels and ground truth labels
    #predicted_labels = []
    #true_labels = []

    #Load the feature extractor
    feature_extractor=load_cnn_feat_ext()

    # Loop through the test dataset and combine bayes filter with CNN predictions
    for i in range(len(test_df)-1):
        #print(i)
        print('Iteration test image number {number}'.format(number=i))

        # Get starting time each iteration -> No, only time the correction step
        #start_time=time.time()

        # Make a movement,from previous (i) to next (i+1) position in test database
        # Predicition step
        print('PREDICTION (movement model)')
        x_mov,y_mov = get_mov(end_pos=test_coords[i+1,:],init_pos=test_coords[i,:])
        u=np.array([x_mov,y_mov])
        bel=get_new_bel(map_coords,bel,u)   # TODO: make faster calculations here

        # Correction step
        # Get starting time in each correction step
        start_time=time.time()

        print('CORRECTION (observation model)')
        # Firstly, do CNN stuff and get most likely label
        img_path = test_df.iloc[i+1, 0]
        print(img_path)
        image = Image.open(img_path).convert('L') #convert to grayscale
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output, probabilities = model(input_tensor)
            probabilities=probabilities.numpy()
        _, predicted = torch.max(output.data, 1)
        predicted_label = predicted.item()
            # prediceted_label gives the most likely cluster
            # probabilities gives the confidence (normalized to 1) that the image is in each cluster
        #print(np.sum(probabilities.numpy()))

        # Then, we get the holistic descriptor from CNN to compare agains others
        input_tensor = transform(image).unsqueeze(0)
        feat=feature_extractor(input_tensor)
        out=feat['lenet_descr']
        test_vector=out.flatten().detach().numpy()
        #print(test_vector)

        # Now we take its most likely cluster and compare with images only from that cluster
        #TODO -> maybe select more than one cluster or none at all
        #print(output)
        #print(probabilities)
        #print(predicted_label)

        print('Predicted CNN label is {label} and it matches with the following from the map'.format(label=predicted_label))


        matching_indices = np.where(map_labels == predicted_label)[0]
        matching_vectors = map_vectors[matching_indices]
        print(matching_indices)
        confidence=probabilities[0,predicted_label]
        print('Confidence: {conf}'.format(conf=confidence))
        
        #print(probabilities)

        # Get the distances from the test vector to each one in the same label
        # Then transform into a normalized probability density function
        distances = cdist([test_vector], matching_vectors, metric='euclidean')
        # min_distance_index = np.argmin(distances)
        # corresponding_index = matching_indices[min_distance_index]
        distances = np.reciprocal(distances)
        distances = distances/np.sum(distances)
        #print(distances)

        # Extend the probability function to have the same size as the map
        prob = np.zeros(len(map_coords))
        prob[matching_indices] = distances
        #print(prob)


        # Update belief
        bel=get_new_bel_correction(bel,prob)       
        #print(bel)

        # Get end time after making all computations
        end_time=time.time()

        # Save result in array and later export to csv
        result[i+1,:]=bel[:]
        times.append(end_time-start_time)
        confidences.append(confidence)


        # # Take the same test image and get data from hog offline test
        # #prob=hog_offline_results[i+1,:] # i+1 -> next position
        # prob=np.reciprocal(prob)
        # prob=np.nan_to_num(prob, posinf=0)

        # #normalize probability vector
        # prob=prob/math.fsum(prob)
        # bel=get_new_bel_correction(bel,prob)


        # # Save result in array and later export to csv
        # result[i+1,:]=bel[:]


    # Compute the confusion matrix
    #conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Create a pandas DataFrame from the matrix and export the DataFrame to a CSV file
    #df = pd.DataFrame(conf_matrix)
    #df.to_csv('alexnet_e100_conf_matrix.csv', index=False, header=False)

    pd.DataFrame(result).to_csv('friburgo_hierarchy_bayes_filter_result.csv', index=None)

    extra_info=np.column_stack((times,confidences))
    pd.DataFrame(extra_info, columns=['cpu time', 'confidence'],).to_csv('friburgo_hierarchy_extra_info.csv', index=None)


if __name__ == "__main__":
    main()
