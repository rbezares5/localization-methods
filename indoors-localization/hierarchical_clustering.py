"""
This script takes a model (map) of the environment from a preamade file and performs spectral clustering, 
allowing to visualize the result and save it for future use
"""


import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import SpectralClustering


# Get model data from csv files
fdTrain=pd.read_csv('GIST_MATLAB_model2.csv', header=0).to_numpy()
mapCoords = pd.read_csv('map_coordinates.csv', header=0).to_numpy()

x=mapCoords[:,0]
y=mapCoords[:,1]

# Spectral Clustering
n_clusters=50
hierarchical_cluster = SpectralClustering(n_clusters=n_clusters, assign_labels='cluster_qr', affinity='nearest_neighbors')
labels = hierarchical_cluster.fit_predict(fdTrain)

# Plot clusters
plt.scatter(x, y, c=labels)
plt.show() 


# Export labels
pd.DataFrame(labels).to_csv("labels_gist50.csv", index=None)