"""
This script impliments a continuous (as opposed to discrete) probabilistic algorithm
based on results from bayes filter

It gives localization estimates outside the discrete positions of the map
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance
import math


map_coords=pd.read_csv('map_coordinates.csv', header=0).to_numpy()
test_coords=pd.read_csv('test_coordinates.csv', header=0).to_numpy()
bayes_results=pd.read_csv('bayes_filter_result.csv', header=0).to_numpy()
n=2 #number of close neighbours

for i in range(len(bayes_results)):
    print(i)
    #https://www.educative.io/answers/how-to-get-indices-of-n-maximum-values-in-numpy-arrays
    # Getting indices of n maximum values
    indices = np.argsort(bayes_results[i,:])[::-1][:n]
    print("Indices:",indices)

    # Getting n maximum values
    prob=bayes_results[i,:][indices]
    print("Values:",prob)

    # Normalize
    prob=prob/math.fsum(prob)
    print("Values (normalized):",prob)

    print('info from closest points in map:')
    x_pred=y_pred=0
    for j in range(n):
        
        idx=indices[j]
        print(prob[j])
        print(map_coords[idx,:])

        x_pred=x_pred+map_coords[idx,0]*prob[j]
        y_pred=y_pred+map_coords[idx,1]*prob[j]

    pose_pred=np.array([x_pred,y_pred])
    print('Prediction vs ground truth:')
    print(pose_pred)
    print(test_coords[i,:])
