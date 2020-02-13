"""
Trajectory of a charged particle influenced by both electric and magnetic field

Related to FYS-4096 Computational Physics
exercise 5 assignments.

By Roman Goncharov on February 2020
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
q = 1
m = 1

"""
Given qE/m = 0.05 and qB/m = 4.00, so we can have q = 1 and m = 1 and
recalculate field magnitudes
"""
def eul(time,dt,E,B):
    """
    Euler's method provided
    time = end time of movement
    dt = step
    E,B = electric and magnetic vector field vectors, respectively
    For details see Week 4 FYS-4096 Computational Physics lecture slides.
    """
    Ex,Ey,Ez = E
    Bx,By,Bz = B
  
    """Initial coordinate r0 = (0,0,0)"""
    x = 0.
    y = 0.
    z = 0.
    
    """Initial velocity v0 = (0.1,0.1,0.1)"""
    vx = 0.1
    vy = 0.1
    vz = 0.1
    
    """Initial time"""
    t = 0.
    
    xx = []
    yy = []
    zz = []

    while t < time:
        x = x + vx * dt
        y = y + vy * dt
        z = z + vz * dt

        vx = vx + (q*Ex/m + (vy*q*Bz/m) - (vz*q*By/m)) * dt
        vy = vy + (q*Ey/m + (vz*q*Bx/m) - (vx*q*Bz/m)) * dt
        vz = vz + (q*Ez/m + (vx*q*By/m) - (vy*q*Bx/m)) * dt
        
        xx.append(x)
        yy.append(y)
        zz.append(z)
        
        t += dt
    print('Coordinates at t = ',t,': ', x,y,z,'Velocity: ',vx,vy,vz)
    return xx,yy,zz


def main():
    """
    All the necessary actions and plotting  perfomed in main()
    """
    E = np.array([0.05,0,0])
    B = np.array([0,4,0])
    xx,yy,zz = eul(5,0.01,E,B)
    

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot3D(xx,yy,zz)
    plt.show()
    
if __name__=="__main__":
    main()
