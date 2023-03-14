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
from sklearn.cluster import SpectralClustering


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


def get_chosen_neighbour_index(map_coords,x,y,j):
    """
    For a given xy point, get the proximity index for the chosen neighbour
    in index j of the map
    """
    # First, make a list of all metric distances
    dist_list=[]
    for k in range(len(map_coords)):
        [x_map,y_map]=map_coords[k][:]
        dist=sqrt((x_map-x)**2+(y_map-y)**2)
        dist_list.append(dist)

    # Then, determine the order of each distance, where 0 is the closest
    index_list=np.zeros(len(dist_list))
    for k in reversed(range(len(dist_list))):
        max_value = max(dist_list)
        index = dist_list.index(max_value)
        index_list[index]=k
        dist_list[index]=0 

    #Finally, return the index corresponding to the prediction made earlier
    return index_list[j]

def plot_neighbours_histogram(neighbour_list):
        # Show plot
        plt.subplot(2,1,1)
        plt.hist(neighbour_list, bins=[0, 50, 100, 200, 300, 400])

        plt.subplot(2,1,2)
        plt.hist(neighbour_list, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        plt.show()

def show_results_from_csv(file):
    """
    Show test results from CSV file
    """
    result=pd.read_csv(file, header=0).to_numpy()
    neighbour=result[:,2]
    d=result[:,0]
    t=result[:,1]

    print('Mean distance error: {value}'.format(value=np.mean(d)))
    print('Mean computing time: {value}'.format(value=np.mean(t)))
    plot_neighbours_histogram(neighbour)


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
        pd.DataFrame(self.map_descriptors).to_csv('{name}_model.csv'.format(name=codename), index=None)

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
        Perform a simulation comparison test images against the map of the environment
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

            # Compute metric distance error between images
            x_test=self.ground_truth[i][0]
            y_test=self.ground_truth[i][1]
            metric_distance=sqrt((x_train-x_test)**2+(y_train-y_test)**2)

            distances.append(metric_distance)
            times.append(end_time-start_time)

            # Get a list of chosen neighbour proximity index, which can later be plotted in a histogram
            index=get_chosen_neighbour_index(self.map_coords,x_test,y_test,min_j)
            neighbour.append(index)

        #self.test_results=zip(distances,times,neighbour)
        self.test_results=np.column_stack((distances,times,neighbour))
        
    def export_test_results(self, codename='DUMMY'):
        """
        Save test results into CSV file for future use
        """
        pd.DataFrame(self.test_results, columns=['distance', 'cpu time', 'neighbour'],).to_csv('batch_location_{name}.csv'.format(name=codename), index=None)

    def import_test_results(self, file):
        """
        Import test results from premade CSV file
        """
        self.test_results=pd.read_csv(file, header=0).to_numpy()

    def show_test_results(self):
        """
        Show test results:
        -Histogram of chosen neighbour proximity index
        -Mean value of distance error and compute time
        """
        d, t, neighbour = zip(*self.test_results)
        
        print('Mean distance error: {value}'.format(value=np.mean(d)))
        print('Mean computing time: {value}'.format(value=np.mean(t)))
        plot_neighbours_histogram(neighbour)

    def get_cluster_labels(self, data, n_clusters=15):
        data=pd.read_csv('GIST_MATLAB_model.csv', header=0).to_numpy() #import external model, could also use internal stored model

        hierarchical_cluster = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', affinity='nearest_neighbors')
        self.labels = hierarchical_cluster.fit_predict(data)

        self.hierarchical_map_descriptors = np.c_[self.labels,self.map_descriptors]   

    def export_cluster_labels(self, codename=''):
        """
        Save representative descriptors into CSV file for future use
        """
        pd.DataFrame(self.labels).to_csv('labels{name}.csv'.format(name=codename), index=None)

    def export_hierarchical_map_descriptors(self, codename='DUMMY'):
        """
        Save map descriptors into CSV file for future use
        """
        pd.DataFrame(self.hierarchical_map_descriptors).to_csv('{name}_hierarchical_model.csv'.format(name=codename), index=None)

    def import_cluster_labels(self, file='labels.csv'):
        """
        Import clustering labels from premade CSV file
        """
        self.labels = pd.read_csv(file, header=0).to_numpy().flatten()
        self.hierarchical_map_descriptors = np.c_[self.labels,self.map_descriptors]

    def plot_clusters(self):
        x=self.map_coords[:,0]
        y=self.map_coords[:,1]
        labels=self.labels

        plt.scatter(x, y, c=labels)
        plt.show() 


    def get_representative_descriptors(self):
        """
        Get the mean representative for each cluster
        """
        representativeVectors=[]
        for label in set(self.labels):
            cluster=self.hierarchical_map_descriptors[self.hierarchical_map_descriptors[:,0]==label]
            repr=cluster.mean(0)
            representativeVectors.append(repr)

        self.representatives=representativeVectors

    def export_representatives(self, codename='DUMMY'):
        """
        Save representative descriptors into CSV file for future use
        """
        pd.DataFrame(self.representatives).to_csv('{name}_representative_descriptors.csv'.format(name=codename), index=None)

    def import_representatives(self, file):
        """
        Import representative descriptors from premade CSV file
        """
        self.representatives=pd.read_csv(file, header=0).to_numpy() 

    def online_hierarchical_hog_test(self, ppc=64):
        distances=[]
        times=[]
        neighbour=[]
        #closest_cluster=[]
        self.contador=0

        # Check all test images
        for i in range(len(self.test_image_files)):
            print(i)
            # Get starting time each iteration
            start_time=time.time()

            # Get image descriptor
            fd, _ = hog(self.test_image_files[i], orientations=8, pixels_per_cell=(ppc, ppc),
                            cells_per_block=(1, 1), visualize=True, channel_axis=-1)
            
            #compare against representatives only to get corresponding cluster (rough location)
            cluster=-1
            minDist=100
            for j in range(len(self.representatives)):
                dist = np.linalg.norm(fd-self.representatives[j][1:]) # Skip first value (label)

                if dist<minDist:
                    minDist=dist
                    cluster=self.representatives[j][0]

            # Compare against model, but only those from the same cluster (fine location)
            min_j=-1
            min_dist=100
            for j in range(len(self.hierarchical_map_descriptors)):
                if self.hierarchical_map_descriptors[j][0]==cluster:
                    dist = np.linalg.norm(fd-self.hierarchical_map_descriptors[j][1:])

                    if dist<min_dist:
                        min_dist=dist
                        min_j=j

            # Get chosen image (prediction) and its xy coords from csv
            [x_train,y_train]=self.map_coords[min_j][:]

            # Get end time after making prediction
            end_time=time.time()

            # Compute metric distance error between images
            x_test=self.ground_truth[i][0]
            y_test=self.ground_truth[i][1]
            metric_distance=sqrt((x_train-x_test)**2+(y_train-y_test)**2)

            distances.append(metric_distance)
            times.append(end_time-start_time)

            # Get a list of chosen neighbour proximity index, which can later be plotted in a histogram
            index=get_chosen_neighbour_index(self.map_coords,x_test,y_test,min_j)
            neighbour.append(index)

            # Get a list of wether it actually chose the closest cluster or not
            self.is_closest_cluster(fd,cluster)
            #closest_cluster.append()

        #self.test_results=zip(distances,times,neighbour)
        self.test_results=np.column_stack((distances,times,neighbour))
        print(self.contador)

    def is_closest_cluster(self,fd,cluster):
        closest_cluster_index=-1
        min_cluster_dist=100

        #print(cluster)
        # Iterate through each cluster
        for i in set(self.labels):
            #print(i)            
            # Compare against model, but only those from "i" cluster
            #min_j=-1
            min_dist_local=100
            for j in range(len(self.hierarchical_map_descriptors)):
                if self.hierarchical_map_descriptors[j][0]==i:
                    dist = np.linalg.norm(fd-self.hierarchical_map_descriptors[j][1:])

                    if dist<min_dist_local:
                        min_dist_local=dist
                        #min_j=j
            
            if min_dist_local<min_cluster_dist:
                min_cluster_dist=min_dist_local
                closest_cluster_index=i

        #print(closest_cluster_index)
        # Check if closest cluster is the chosen one
        if closest_cluster_index==cluster:
            print(True)
            self.contador+=1



