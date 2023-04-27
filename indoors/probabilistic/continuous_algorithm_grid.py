"""
This script impliments a continuous (as opposed to discrete) probabilistic algorithm
based on results from bayes filter, modified for a grid-based map

It gives localization estimates outside the discrete positions of the map
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance
import math
from statistics import mean


def get_prediction(Nx, Ny, cell_size, ground_truth, bayes_results, n):
    pose_pred=[]
    error=[]
    for j in range(len(bayes_results)):
        indices = np.argsort(bayes_results[j,:])[::-1][:n]
        print(indices)

        # Getting n maximum values
        prob=bayes_results[j,:][indices]
        print("Values:",prob)

        # Normalize
        prob=prob/math.fsum(prob)
        print("Values (normalized):",prob)

        # print(np.unravel_index(indices,(Nx,Ny)))
        loc=np.unravel_index(indices,(Nx,Ny))
        #print(loc)
        x_pred=y_pred=0
        for i in range(n):
            # print(indices[i])
            # print(bayes_results[0,indices[i]])
            # print(loc[0][i]*cell_size,loc[1][i]*cell_size)
            
            x_pred=x_pred+loc[0][i]*cell_size*prob[i]
            y_pred=y_pred+loc[1][i]*cell_size*prob[i]

        pose_pred.append(np.array([x_pred,y_pred]))
        #pose_pred=np.array([x_pred,y_pred])
        print('Prediction vs ground truth:')
        print(pose_pred[j])
        print(ground_truth[j,:])
        #Serror=distance.euclidean(pose_pred[j],ground_truth[j,:])
        error.append(distance.euclidean(pose_pred[j],ground_truth[j,:]))
        #print(error)
    return pose_pred,error


def main():
    accesible_coords=pd.read_csv('Qevent_map_coordinates.csv', header=0).to_numpy()
    ground_truth=pd.read_csv('Qevent_test_coordinates.csv', header=0).to_numpy()
    bayes_results=pd.read_csv('Qevent_bayes_filter_result2.csv', header=0).to_numpy()

    cell_size=40
    Nx=int(np.max(accesible_coords[:,0])/cell_size)
    Ny=int(np.max(accesible_coords[:,1])/cell_size)



    p1,e1=get_prediction(Nx,Ny,cell_size,ground_truth,bayes_results,n=1)
    p2,e2=get_prediction(Nx,Ny,cell_size,ground_truth,bayes_results,n=2)
    p3,e3=get_prediction(Nx,Ny,cell_size,ground_truth,bayes_results,n=3)
    p4,e4=get_prediction(Nx,Ny,cell_size,ground_truth,bayes_results,n=4)

    error=np.column_stack((e1,e2,e3,e4))

    print('Mean error (n=1): {e}'.format(e=mean(e1)))
    print('Mean error (n=2): {e}'.format(e=mean(e2)))
    print('Mean error (n=3): {e}'.format(e=mean(e3)))
    print('Mean error (n=4): {e}'.format(e=mean(e4)))

    pd.DataFrame(error, columns=['n=1', 'n=2', 'n=3', 'n=4']).to_csv('Qevent_continuous_algorithm_error.csv', index=None)
    

if __name__ == "__main__":
    main()