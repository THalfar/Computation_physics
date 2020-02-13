# -*- coding: utf-8 -*-
"""
Calculating the trajectory of a charged paricle in electric and magnetig
fields.
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def calculate_trajectory(r, v, t_start, t_end, F):
    # A function that calculates the trajectory of a charged particle in an
    # electric field.
    
    # Initializing the arrays for the trajectory and velocity
    dt = 0.0001
    N = int((t_end - t_start)/dt)
    
    points = np.zeros((N, 3))
    points[0] = r
    velocities = np.zeros((N, 3))
    velocities[0] = v
    
    # Calculate the other location and velocity values.
    for i in range(N-1):
        points[i + 1] = points[i] + velocities[i]*dt
        velocities[i + 1] = velocities[i] + F(velocities[i])*dt

    return points, velocities      

def main():
    
    # Defining the starting parameters.
    t_start = 0
    t_end = 5
    r = [0.0, 0.0, 0.0]
    v = [0.1, 0.1, 0.1]
    state = np.zeros((2,3))
    state[0] = r
    state[1] = v
    
    # Defining the force field.
    def F(v): return [0.05, 0, 0] + np.cross(v, [0, 4.0, 0])
    
    # Calculating the trajectory and velocities
    trajectory, velocity = calculate_trajectory(r, v, t_start, t_end, F)
    
    # Plot the trajectory.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2])
    # Print the end values
    print("At t = 5:")
    print("r =", trajectory[-1])
    print("v =", velocity[-1])

    
if __name__=="__main__":
    main()