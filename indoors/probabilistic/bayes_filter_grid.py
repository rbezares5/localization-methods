import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import random

class MarkovLocalization:
    def __init__(self, map, Nx, Ny, motion_model, sensor_model, prior=None, cell_size=1):
        #self.map_size = map_size
        self.map = map
        self.motion_model = motion_model
        self.sensor_model = sensor_model
        self.cell_size = cell_size
        self.Nx=Nx
        self.Ny=Ny
        self.belief = np.zeros((self.Nx,self.Ny))
        self.prior = prior if prior is not None else np.ones((self.Nx,self.Ny)) / (self.Nx * self.Ny)

    def update(self, observation, control):
        self.belief = self.motion_model.apply(self.belief, control, self.Nx, self.Ny, self.cell_size)
        self.belief = self.sensor_model.apply(self.belief, observation, self.Nx, self.Ny, self.cell_size)

    def get_estimate(self):
        return self.belief

    def get_most_likely_position(self):
        return np.unravel_index(np.argmax(self.belief), self.belief.shape)

    def set_prior(self, prior):
        self.prior = prior

    def reset_belief(self):
        self.belief = self.prior

     
class GridMotionModel:
    def __init__(self, noise_std):
        self.noise_std = noise_std

    def apply(self, belief, control, Nx, Ny, cell_size):
        # Compute the transition matrix based on the control input and current pose
        #T = self.compute_transition_matrix(control_x, control_y, map_size, cell_size)
        T = self.compute_T_gauss()

        # Update the belief by multiplying with the transition matrix
        #new_belief = T @ belief.flatten()
        #print(T)

        control_x, control_y = control
        control_x /= cell_size
        control_y /=+ cell_size
        new_belief = np.zeros_like(belief)
        #x1, y1 = np.unravel_index(np.argmax(belief), belief.shape)

        for k in range(Nx*Ny):
            x1, y1 = np.unravel_index(k, belief.shape)
            x2, y2 = x1 + control_x, y1 + control_y

            # Check if destination is out of bounds
            #x2 = max(0, min(x2, Nx - 1))
            #y2 = max(0, min(y2, Ny - 1))
            #new_belief = np.zeros_like(belief)

            bel=np.zeros_like(belief)
            for i in range(belief.shape[0]):
                for j in range(belief.shape[1]):
                    dist=np.sqrt((i-x2)**2 + (j-y2)**2)
                    p=np.exp(-dist**2/(2*self.noise_std**2))
                    bel[i,j]=p

            bel = convolve(bel,T)
            bel *= belief[x1, y1]
        #new_belief = new_belief.reshape(belief.shape)

            new_belief += bel
        # Add noise and normalize the belief
        #new_belief = self.add_noise(new_belief)
        new_belief /= np.sum(new_belief)

        return new_belief
    
    def compute_T_gauss(self):
        noise_std=0.5
        size_T=3
        center=1
        T=np.zeros((size_T,size_T))
        for i in range(size_T):
            for j in range(size_T):
                dist=np.sqrt((i-center)**2 + (j-center)**2)
                p=np.exp(-dist**2/(2*noise_std**2))
                T[i,j]=p
        T=T/np.sum(T)

        return T

    def add_noise(self, belief):
        noise = np.random.normal(0.0, self.noise_std, belief.shape)
        noisy_belief = belief + noise
        return np.clip(noisy_belief, 0.0, 1.0)
    
class HogSensorModel:
    def __init__(self, hog_offline_results, accesible_coords):
        #self.noise_std = noise_std
        self.hog_offline_results = hog_offline_results
        self.accesible_coordinates = accesible_coords

    def apply(self, belief, observation, Nx, Ny, cell_size):
        fdp_hog=np.zeros((Nx,Ny))
        fdp=np.reciprocal(observation)
        for i in range(Nx):
            for j in range(Ny):
                if [i*cell_size,j*cell_size] in self.accesible_coordinates.tolist():
                    idx=self.accesible_coordinates.tolist().index([i*cell_size,j*cell_size])             
                    fdp_hog[i][j]=fdp[idx]

        #fdp_hog /= np.sum(fdp_hog)
        new_belief = belief*fdp_hog
        new_belief /= np.sum(new_belief)
        #print(np.sum(bel))    
        return new_belief

def plot(bel, Nx, Ny, s=1000):
    # Assume that `belief` is your numpy array containing the belief probabilities
    # `map_size` is the size of the map in terms of grid cells

    # Create a list of x,y coordinates and sizes based on the belief values
    xy = []
    sizes = []
    for i in range(Nx):
        for j in range(Ny):
            size = s * bel[i,j] # Scale up the size for better visualization
            if size > 0:
                #xy.append((j, i)) # Note the order (j, i) instead of (i, j) to match array indexing
                xy.append((i, j))
                sizes.append(size)

    # Plot the scatter plot using the x,y coordinates and sizes
    plt.scatter(*zip(*xy), s=sizes)
    #plt.scatter(*zip(*xy), c=sizes)

    # Set the axis limits to match the map size
    plt.xlim(0, Nx)
    plt.ylim(0, Ny)

    plt.colorbar()
    # Show the plot
    plt.show()

def get_mov(end_pos,init_pos,noise_std=0.1):
    noise=random.gauss(0,noise_std)
    x_mov=end_pos[0]-init_pos[0]+noise
    y_mov=end_pos[1]-init_pos[1]+noise

    return x_mov,y_mov

def main():
    # Load data from CSV
    accesible_coords=pd.read_csv('Qevent_map_coordinates.csv', header=0).to_numpy()
    ground_truth=pd.read_csv('Qevent_test_coordinates.csv', header=0).to_numpy()
    hog_offline_results=pd.read_csv('Qevent_hog_descriptor_distances.csv', header=0).to_numpy()
    cell_size=40
    Nx=int(np.max(accesible_coords[:,0])/cell_size)
    Ny=int(np.max(accesible_coords[:,1])/cell_size)

    # Create matrix of blocked (0) and accesible (1) positions 1
    blocked_coords=np.zeros((Nx,Ny))
    for i in range(Nx):
        for j in range(Ny):
            if [i*cell_size,j*cell_size] in accesible_coords.tolist():
                blocked_coords[i][j]=1

    # Initialize belief around first test coordinate
    initial_bel=np.zeros((Nx,Ny))
    x2=660/cell_size
    y2=60/cell_size
    noise_std=0.5
    for i in range(initial_bel.shape[0]):
        for j in range(initial_bel.shape[1]):
            dist=np.sqrt((i-x2)**2 + (j-y2)**2)
            p=np.exp(-dist**2/(2*noise_std**2))
            initial_bel[i,j]=p
    initial_bel=initial_bel/np.sum(initial_bel)


    # Initilize simulation objects and set initial belief
    mov_model=GridMotionModel(0.7)
    obs_model=HogSensorModel(hog_offline_results, accesible_coords)
    localization=MarkovLocalization(blocked_coords,Nx,Ny,mov_model,obs_model,cell_size=cell_size)
    localization.set_prior(initial_bel)
    localization.reset_belief()


    print(localization.get_most_likely_position())
    bel=localization.get_estimate()
    print(np.sum(bel))
    plot(bel,Nx,Ny)

    result=np.zeros((len(ground_truth),Nx*Ny))
    result[0,:]=bel.flatten()
    # After initial belief, compute the rest on loop
    for k in range(1, len(ground_truth)):
        # Get control command
        x_mov,y_mov = get_mov(end_pos=ground_truth[k,:],init_pos=ground_truth[k-1,:])
        u=np.array([x_mov,y_mov])

        # Get observation
        z=np.reciprocal(hog_offline_results[k,:])

        # Apply prediction and correction
        localization.update(control=u, observation=z)

        # Get results and show onscreen
        print(localization.get_most_likely_position())
        bel=localization.get_estimate()
        print(np.sum(bel))
        plot(bel,Nx,Ny)

        # Save result in array and later export to csv
        result[k,:]=bel.flatten()

    pd.DataFrame(result).to_csv('Qevent_bayes_filter_result2.csv', index=None)


if __name__ == "__main__":
    main()


