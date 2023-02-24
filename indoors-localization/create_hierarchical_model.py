"""
This script takes the model (map) of the environment created using the HOG descriptor
and creates a hierarchical model using spectral clustering
"""

#create hierarchical map from first model
import numpy as np
from sklearn.cluster import  SpectralClustering
import pandas as pd

def main():
    #get model data from csv files
    fdTrain=pd.read_csv("HOG_model.csv", header=0).to_numpy()
    mapCoords = pd.read_csv("model_coordinates.csv", header=0).to_numpy()

    n_clusters=5
    hierarchical_cluster = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', affinity='nearest_neighbors')
    labels = hierarchical_cluster.fit_predict(fdTrain)

    #append labels column to train descriptor data
    fdTrainClusters=np.c_[labels,fdTrain]
    #print(fdTrainClusters)

    representativeVectors=[]
    for label in set(labels):

        print(label)


        cluster=fdTrainClusters[fdTrainClusters[:,0]==label]
        repr=cluster.mean(0)
        print(repr)

        representativeVectors.append(repr)

    #print(representativeVectors)

    #save result data in csv file
    pd.DataFrame(fdTrainClusters).to_csv("HOG_hierarchical_model.csv", index=None)
    pd.DataFrame(representativeVectors).to_csv("HOG_representative_descriptors.csv", index=None)

if __name__ == "__main__":
    main()    