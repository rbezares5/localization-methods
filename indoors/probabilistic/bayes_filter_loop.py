import pandas as pd
import numpy as np
from scipy.spatial import distance
import random
import math

def get_mov(end_pos,init_pos,noise_deviation=0.1):
    noise=random.gauss(0,noise_deviation)
    x_mov=end_pos[0]-init_pos[0]+noise
    y_mov=end_pos[1]-init_pos[1]+noise

    return x_mov,y_mov

def movement_model(map,starting_pos,u,threshold=1):
    """
    This function gives the probabilty of landing at each spot from a given spot
    """
    prob=np.zeros(len(map))

    mov=starting_pos+u

    #get closest distances in map
    for i in range(len(map)):
    
        dist=distance.euclidean(map[i,:],mov)
        if dist<threshold:
            prob[i]=dist 


    #get inverse proportional values to distance
    prob=np.reciprocal(prob)
    prob=np.nan_to_num(prob, posinf=0)


    #normalize probability vector
    prob=prob/math.fsum(prob)

    return prob

def get_new_bel(map,bel,u):
    new_bel=np.zeros(len(map))

    # For a given movement "u", get the probability of landing in each of the map's positions
    for i in range(len(map)):
        prob=movement_model(map,starting_pos=map[i,:],u=u)

        new_bel[i]=math.fsum(bel*prob)

    # Correct belief vector so its sum of elements amount to 1 (add difference to highest probability)
    diff=1-math.fsum(new_bel)
    index=np.argmax(new_bel)
    new_bel[index]+=diff

    return new_bel

def get_new_bel_correction(bel,prob):
    new_bel=bel*prob
    new_bel=new_bel/math.fsum(new_bel)

    return new_bel

def main():

    # Initialize external data
    map_coords=pd.read_csv('map_coordinates.csv', header=0).to_numpy()
    test_coords=pd.read_csv('test_coordinates.csv', header=0).to_numpy()
    hog_offline_results=pd.read_csv('hog_descriptor_distances.csv', header=0).to_numpy()

    # Get initial belief
    bel=np.full(len(map_coords),fill_value=1/len(map_coords))
    bel=movement_model(map=map_coords,starting_pos=test_coords[0,:],u=[0,0])


    # Make it loop
    result=np.zeros((len(test_coords)-1,len(map_coords)))
    for i in range(len(test_coords)-1):
        print('Iteration test image number {number}'.format(number=i))

        # Make a movement,from previous (i) to next (i+1) position in test database

        # Predicition step
        print('PREDICTION (movement model)')
        x_mov,y_mov = get_mov(end_pos=test_coords[i+1,:],init_pos=test_coords[i,:])
        u=np.array([x_mov,y_mov])

        bel=get_new_bel(map_coords,bel,u)   # TODO: make faster calculations here


        # Correction step
        print('CORRECTION (observation model)')
        # Take the same test image and get data from hog offline test
        prob=hog_offline_results[i+1,:] # i+1 -> next position
        prob=np.reciprocal(prob)
        prob=np.nan_to_num(prob, posinf=0)

        #normalize probability vector
        prob=prob/math.fsum(prob)
        bel=get_new_bel_correction(bel,prob)


        # Save result in array and later export to csv
        result[i,:]=bel[:]

    pd.DataFrame(result).to_csv('bayes_filter_result.csv', index=None)


if __name__ == "__main__":
    main()