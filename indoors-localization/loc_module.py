"""
This module implements python classes and related functions to solve the localization problem
"""

import cv2 as cv
from skimage.feature import hog
import glob
from natsort import natsorted
import pandas as pd
import numpy as np
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

def show_results(file):
    """
    Show test results from CSV file
    """
    result=pd.read_csv(file, header=0).to_numpy()

    neighbour=result[:,2]
    print(neighbour)
    plt.subplot(2,1,1)
    plt.hist(neighbour, bins=[0, 50, 100, 200, 300, 400])

    plt.subplot(2,1,2)
    plt.hist(neighbour, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.show()

class EnvironmentModel:
    def __init__(self, train_dataset_path):
        self.train_image_files = train_dataset_path


    def get_map_coords(self):
        """
        Get coordinates of images located in train_dataset_path
        """
        self.map_coords=[]
        for file in natsorted(glob.glob(self.train_image_files)):
            x,y=get_xy(file)
            self.map_coords.append([x,y])

    
    def export_map_coords(self):
        """
        Save map coordinates into CSV file for future use
        """
        pd.DataFrame(self.map_coords, columns=['x', 'y'],).to_csv('map_coordinates.csv', index=None)


    def import_map_coords(self, file):
        """
        Import map coordinates from premade CSV file
        """
        self.map_coords=pd.read_csv(file, header=0).to_numpy()

    def create_hog_map(self, ppc=64):
        """
        Create a map of the environment using the HOG descriptor
        """
        # Get all train images
        train_images=[]
        for file in natsorted(glob.glob(self.train_image_files)):
            train_images.append(cv.imread(file)) #could use PIL instead of cv?

        # Get hog descriptor of each image
        self.map_descriptors=[]
        for i in range(len(train_images)):
            print(i)
            fd, _ = hog(train_images[i], orientations=8, pixels_per_cell=(ppc, ppc),
                            cells_per_block=(1, 1), visualize=True, channel_axis=-1)
            
            self.map_descriptors.append(fd)

    def export_map_descriptors(self, codename='DUMMY'):
        """
        Save map descriptors into CSV file for future use
        """
        pd.DataFrame(self.map_descriptors).to_csv("{name}_model.csv".format(name=codename), index=None)

    def import_map_descriptors(self, file):
        """
        Import map descriptors from premade CSV file
        """
        self.map_descriptors=pd.read_csv(file, header=0).to_numpy()

    def import_test_images(self, test_dataset_path):
        """
        Get filenames of images located in test_dataset_path
        """
        self.test_image_files=[]
        self.ground_truth=[]
        for file in natsorted(glob.glob(test_dataset_path)):
            self.test_image_files.append(cv.imread(file))
            x,y=get_xy(file)
            self.ground_truth.append([x,y])

    def online_hog_test(self, ppc=64):
        """
        Perform a simulation comparion test images against the map of the environment
        Both should use the same descriptor (HOG)
        """
        distances=[]
        times=[]
        neighbour=[]

        # Check all test images
        for i in range(len(self.test_image_files)):
            print(i)
            # Get starting time each iteration
            start_time=time.time()

            # Get image descriptor
            fd, _ = hog(self.test_image_files[i], orientations=8, pixels_per_cell=(ppc, ppc),
                            cells_per_block=(1, 1), visualize=True, channel_axis=-1)

            # Compare against model
            min_j=-1
            min_dist=100
            for j in range(len(self.map_descriptors)):
                dist = np.linalg.norm(fd-self.map_descriptors[j])

                if dist<min_dist:
                    min_dist=dist
                    min_j=j

            # Get chosen image (prediction) and its xy coords from csv
            [x_train,y_train]=self.map_coords[min_j][:]

            # Get end time after making prediction
            end_time=time.time()

            # Compute error metric distance bteween images
            x_test=self.ground_truth[i][0]
            y_test=self.ground_truth[i][1]
            metric_distance=sqrt((x_train-x_test)**2+(y_train-y_test)**2)

            distances.append(metric_distance)
            times.append(end_time-start_time)

            #chosen neighbor histogram list -> this should be a separate function
            #first, make a list of all metric distances
            dist_list=[]
            for k in range(len(self.map_descriptors)):
                [x_map,y_map]=self.map_coords[k][:]
                dist=sqrt((x_map-x_test)**2+(y_map-y_test)**2)
                dist_list.append(dist)

            #then, determine the order of each distance, where 0 is the closest
            index_list=np.zeros(len(dist_list))
            for k in reversed(range(len(dist_list))):
                max_value = max(dist_list)
                index = dist_list.index(max_value)
                index_list[index]=k
                dist_list[index]=0

            #get the index corresponding to the prediction made earlier
            neighbour.append(index_list[min_j])

        self.test_results=zip(distances,times,neighbour)
        

    def export_test_results(self, codename='DUMMY'):
        """
        Save test results into CSV file for future use
        """
        pd.DataFrame(self.test_results, columns=['distance', 'cpu time', 'neighbour'],).to_csv("batch_location_{name}.csv".format(name=codename), index=None)

    def import_test_results(self, file):
        """
        Import test results from premade CSV file
        """
        self.test_results=pd.read_csv(file, header=0).to_numpy()    

    def show_neighbours_histogram(self):
        """
        Show histogram of chosen neighbour from test results
        """
        d, t, neighbour = zip(*self.test_results)
        
        # Show plot
        plt.subplot(2,1,1)
        plt.hist(neighbour, bins=[0, 50, 100, 200, 300, 400])

        plt.subplot(2,1,2)
        plt.hist(neighbour, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        plt.show()
