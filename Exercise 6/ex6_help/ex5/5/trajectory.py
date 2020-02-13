""" 
---------- EXERCISE 5 - problem 2 -----------
----- FYS-4096 - Computational Physics ------

Calculates the path of a charged particle in a field
with given effects

:function: trajectory: calculates the path and velocity 
                       of a charged particle in a magnetic
                       and electric field then plots both
"""
import numpy as np
from runge_kutta import runge_kutta4
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def trajectory(rv,t,tmax,dt, E, B):
    """
    Recursively solves the path of a charged particle trajectory.
    :param: rv     three dimensional matrix that has all positions and
                   velocities for each timestep
    :param: t      time at calculation
    :param: tmax   time to end the calculation
    :param: dt     size of the timestep
    :param: E      the effect-vector of electric field (qE/m)
    :param: B      the effect-vector of magnetic field (qB/m)
    
    :return: rv-matrix or trajectory function itself
    """
    
    # nested function definition for the equation of motion for the particle
    def lorentz(xf, tf): return np.array([xf[1,:], E + np.cross(xf[1,:],B)])
    
    # solving the value at next time step
    x, t = runge_kutta4(rv[-1,:,:],t,dt,lorentz)
    
    # append the new velocity and location
    rv = np.append(rv, [x], axis=0)
    
    # if tmax is not reached yet return functino recursively
    if t > tmax-dt:
        return rv
    
    # otherwise return the trajectory matrix 
    return trajectory(rv,t,tmax,dt,E,B)


def main():
    
    # initialize the starting conditions
    t0 = 0
    tmax = 5
    dt = 0.1
    
    # effects of the electric and magnetic fields on the charged particle
    # defined as q|E|/m = 0.05 and q|B|/m = 4.0
    E = np.array([0.05,0.0,0.0])
    B = np.array([0.0,4.0,0.0])
    
    # first element in the location/velocity matrix
    # location is 0,0,0 and velocity is 0.1,0.1,0.1
    rv0 = np.array([[[0.0,0.0,0.0],[0.1,0.1,0.1]]])
    
    # call the function
    rv = trajectory(rv0,t0,tmax,dt,E,B)
    
    # plot path
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(rv[:,0,0],rv[:,0,1],rv[:,0,2])
    
    # and velocity
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(rv[:,1,0],rv[:,1,1],rv[:,1,2])
    
    plt.show()

if __name__ == "__main__":
    main()
