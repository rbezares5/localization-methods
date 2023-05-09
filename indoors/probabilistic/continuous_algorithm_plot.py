import pandas as pd
import numpy as np
from scipy.spatial import distance
import math
from statistics import mean
import matplotlib.pyplot as plt

def main():
    #map_coords=pd.read_csv('map_coordinates.csv', header=0).to_numpy()
    test_coords=pd.read_csv('Qevent_test_coordinates.csv', header=0).to_numpy()
    #bayes_results=pd.read_csv('bayes_filter_result.csv', header=0).to_numpy()

    error=pd.read_csv('Qevent_continuous_algorithm_error.csv', header=0).to_numpy()

    print(error)
    print(error[0,:])
    print(len(error))

    indices=np.argmin(error,axis=1)
    print(indices)
    print(len(indices))

    x=test_coords[:,0]
    y=test_coords[:,1]

    plt.scatter(x, y, c=indices)
    #plt.legend(['1','2','3'])
    plt.colorbar()
    plt.show() 


if __name__ == "__main__":
    main()