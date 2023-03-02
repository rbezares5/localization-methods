"""
This script takes the model (map) of the environment created using a descriptor (ex: HOG, CNN)
and creates a hierarchical model using spectral clustering
"""

#create hierarchical map from first model
import numpy as np
from sklearn.cluster import  SpectralClustering
import pandas as pd

def main():
    #get model data from csv files
    fdTrain=pd.read_csv("HOG_model.csv", header=0).to_numpy()
    #fdTrain=pd.read_csv('Alexnet_c4_model.csv', header=0).to_numpy()
    #fdTrain=pd.read_csv('Alexnet_c5_model.csv', header=0).to_numpy()

    # Generate clusters
    # n_clusters=10
    # hierarchical_cluster = SpectralClustering(n_clusters=n_clusters, assign_labels='cluster_qr', affinity='nearest_neighbors')
    # labels = hierarchical_cluster.fit_predict(fdTrain)

    # Import clusters from external file
    labels=pd.read_csv('labels.csv', header=0).to_numpy().flatten()
        # Cluster labels obtained using gist descriptor in MATLAB
    
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
    pd.DataFrame(fdTrainClusters).to_csv("NEW_hierarchical_model.csv", index=None)
    pd.DataFrame(representativeVectors).to_csv("NEW_representative_descriptors.csv", index=None)

if __name__ == "__main__":
    main()    