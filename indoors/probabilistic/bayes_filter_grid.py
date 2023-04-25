import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

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
        #self.belief = self.sensor_model.apply(self.belief, observation, self.map_size)

    def get_estimate(self):
        return self.belief

    def get_most_likely_position(self):
        return np.unravel_index(np.argmax(self.belief), self.belief.shape)

    def set_prior(self, prior):
        self.prior = prior

    def reset_belief(self):
        self.belief = self.prior


class StraightLineMotionModel:
    def __init__(self, noise_std):
        self.noise_std = noise_std

    def apply(self, belief, control, map_size):
        control_x, control_y = control
        x1, y1 = np.unravel_index(np.argmax(belief), belief.shape)
        x2, y2 = x1 + control_x, y1 + control_y
        x2 = max(0, min(x2, map_size - 1))
        y2 = max(0, min(y2, map_size - 1))
        new_belief = np.zeros_like(belief)
        #new_belief[x2, y2] = 1.0
        for i in range(belief.shape[0]):
            for j in range(belief.shape[1]):
                dist=np.sqrt((i-x2)**2 + (j-y2)**2)
                p=np.exp(-dist**2/(2*self.noise_std**2))
                new_belief[i,j]=p
        #new_belief = self.add_noise(new_belief)
        #Normalize
        new_belief = new_belief/np.sum(new_belief)

        return new_belief

    def add_noise(self, belief):
        noise = np.random.normal(0.0, self.noise_std, belief.shape)
        noisy_belief = belief + noise
        return np.clip(noisy_belief, 0.0, 1.0)

class GridMotionModel:
    def __init__(self, noise_std):
        self.noise_std = noise_std

    def apply(self, belief, control, map_size, cell_size):
        control_x, control_y = control
        x1, y1 = np.unravel_index(np.argmax(belief), belief.shape)

        # Compute the transition matrix based on the control input and current pose
        T = self.compute_transition_matrix(control_x, control_y, map_size, cell_size)


        # Update the belief by multiplying with the transition matrix
        new_belief = T @ belief.flatten()

        print(T)

        new_belief = new_belief.reshape(belief.shape)

        # Add noise and normalize the belief
        #new_belief = self.add_noise(new_belief)
        new_belief /= np.sum(new_belief)

        return new_belief

    def compute_transition_matrix(self, control_x, control_y, map_size, cell_size):
        # Compute the transition matrix based on the control input and current pose
        T = np.zeros((map_size**2, map_size**2))
        for i in range(map_size):
            for j in range(map_size):
                idx = np.ravel_multi_index((i,j), (map_size,map_size))
                if idx < 0 or idx >= map_size**2:
                    continue
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if abs(dx) + abs(dy) > 1:
                            continue
                        if i + dx < 0 or i + dx >= map_size or j + dy < 0 or j + dy >= map_size:
                            continue
                        idx2 = np.ravel_multi_index((i+dx, j+dy), (map_size,map_size))
                        p = self.compute_probability(i,j,i+dx,j+dy,control_x,control_y,cell_size)
                        T[idx, idx2] = p
        return T

    def compute_probability(self, x1, y1, x2, y2, control_x, control_y, cell_size):
        # Compute the probability of transitioning from cell (x1,y1) to cell (x2,y2)
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2) * cell_size
        angle = np.arctan2(y2-y1, x2-x1) - np.arctan2(control_y, control_x)
        angle = np.mod(angle + np.pi, 2*np.pi) - np.pi
        p_dist = np.exp(-dist**2 / (2*self.noise_std**2))
        p_angle = np.exp(-angle**2 / (2*(np.pi/4)**2))
        p = p_dist * p_angle
        return p_dist

    def add_noise(self, belief):
        noise = np.random.normal(0.0, self.noise_std, belief.shape)
        noisy_belief = belief + noise
        return np.clip(noisy_belief, 0.0, 1.0)
        
class GridMotionModel2:
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
        control_y /= cell_size
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

    def apply(self, belief, control, Nx, Ny, cell_size):
        fdp_hog=np.zeros((Nx,Ny))
        fdp=np.reciprocal(self.hog_offline_results[i,:])
        for i in range(Nx):
            for j in range(Ny):
                if [i*cell_size,j*cell_size] in self.accesible_coordinates.tolist():
                    idx=self.accesible_coordinates.tolist().index([i*cell_size,j*cell_size])             
                    fdp_hog[i][j]=fdp[idx]

        #fdp_hog /= np.sum(fdp_hog)
        new_belief = new_belief*fdp_hog
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

def main():
    accesible_coords=pd.read_csv('Qevent_map_coordinates.csv', header=0).to_numpy()
    ground_truth=pd.read_csv('Qevent_test_coordinates.csv', header=0).to_numpy()
    hog_offline_results=pd.read_csv('Qevent_hog_descriptor_distances.csv', header=0).to_numpy()
    cell_size=40
    Nx=int(np.max(accesible_coords[:,0])/cell_size)
    Ny=int(np.max(accesible_coords[:,1])/cell_size)
    blocked_coords=np.zeros((Nx,Ny))

    for i in range(Nx):
        for j in range(Ny):
            if [i*cell_size,j*cell_size] in accesible_coords.tolist():
                blocked_coords[i][j]=1

    #plot(blocked_coords, i, j, 100)

    # fdp_hog=np.zeros((Nx,Ny))
    # fdp=np.reciprocal(hog_offline_results[0,:])
    # for i in range(Nx):
    #     for j in range(Ny):
    #         if [i*cell_size,j*cell_size] in accesible_coords.tolist():
    #             idx=accesible_coords.tolist().index([i*cell_size,j*cell_size])
    #             #print(idx)
                
    #             fdp_hog[i][j]=fdp[idx]

    # #fdp_hog /= np.sum(fdp_hog)
    # fdp_hog /= np.max(fdp_hog)
    # plot(fdp_hog, i, j, 100)


    initial_bel=np.zeros((Nx,Ny))
    #initial_bel[2][2]=1
    x2=660/cell_size
    y2=60/cell_size
    noise_std=0.5
    for i in range(initial_bel.shape[0]):
        for j in range(initial_bel.shape[1]):
            dist=np.sqrt((i-x2)**2 + (j-y2)**2)
            p=np.exp(-dist**2/(2*noise_std**2))
            initial_bel[i,j]=p
    initial_bel=initial_bel/np.sum(initial_bel)
    #mov_model=StraightLineMotionModel(0.7)
    #blocked=np.zeros((N,N))
    mov_model=GridMotionModel2(0.7)
    localization=MarkovLocalization(blocked_coords,Nx,Ny,mov_model,mov_model,cell_size=cell_size)
    localization.set_prior(initial_bel)
    localization.reset_belief()


    print(localization.get_most_likely_position())
    bel=localization.get_estimate()
    #print(bel)
    print(np.sum(bel))
    plot(bel,Nx,Ny)

    for i in range(1, len(ground_truth)):
        #u=[1,3]
        #u=[20,110]
        #Prediccion
        print('PREDICCION')
        u=[ground_truth[i][0]-ground_truth[i-1][0],ground_truth[i][1]-ground_truth[i-1][1]]
        localization.update(observation=[0,0],control=u)
        print(localization.get_most_likely_position())
        bel=localization.get_estimate()
        #print(bel)
        print(np.sum(bel))

        #CORRECCION
        print('CORRECCION')
        fdp_hog=np.zeros((Nx,Ny))
        fdp=np.reciprocal(hog_offline_results[i,:])
        for i in range(Nx):
            for j in range(Ny):
                if [i*cell_size,j*cell_size] in accesible_coords.tolist():
                    idx=accesible_coords.tolist().index([i*cell_size,j*cell_size])             
                    fdp_hog[i][j]=fdp[idx]

        fdp_hog /= np.max(fdp_hog)
        #fdp_hog /= np.sum(fdp_hog)
        bel=bel*fdp_hog
        bel /= np.sum(bel)
        print(np.sum(bel))



        plot(bel,Nx,Ny)



if __name__ == "__main__":
    main()


