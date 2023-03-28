"""
This script impliments a continuous (as opposed to discrete) probabilistic algorithm
based on results from bayes filter

It gives localization estimates outside the discrete positions of the map
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance
import math
from statistics import mean



def get_prediction(map_coords, test_coords, bayes_results, n):
    pose_pred=[]
    error=[]
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

        #pose_pred=np.array([x_pred,y_pred])
        pose_pred.append(np.array([x_pred,y_pred]))
        print('Prediction vs ground truth:')
        print(pose_pred[i])
        print(test_coords[i,:])
        error.append(distance.euclidean(pose_pred[i],test_coords[i,:]))
    
    return pose_pred,error


def main():
    map_coords=pd.read_csv('BIS_map_coordinates.csv', header=0).to_numpy()
    test_coords=pd.read_csv('test_coordinates.csv', header=0).to_numpy()
    bayes_results=pd.read_csv('BIS_bayes_filter_result.csv', header=0).to_numpy()

    p1,e1=get_prediction(map_coords,test_coords,bayes_results,n=1)
    p2,e2=get_prediction(map_coords,test_coords,bayes_results,n=2)
    p3,e3=get_prediction(map_coords,test_coords,bayes_results,n=3)

    error=np.column_stack((e1,e2,e3))

    print('Mean error (n=1): {e}'.format(e=mean(e1)))
    print('Mean error (n=2): {e}'.format(e=mean(e2)))
    print('Mean error (n=3): {e}'.format(e=mean(e3)))

    pd.DataFrame(error, columns=['n=1', 'n=2', 'n=3']).to_csv('BIS_continuous_algorithm_error.csv', index=None)
    


if __name__ == "__main__":
    main()