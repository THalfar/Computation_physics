# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:15:47 2020

@author: halfar
"""

from runge_kutta import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation




def particle_world():
    
    E = np.array([0.05, 0, 0])
    B = np.array([0, 4.0, 0])
  
    def step(state,t, E, B):
        
        # separate to two diff.eq.
        # speed
        drdt = state[3:] #last three are velocity x,y,z values
        # acceleration 
        dvdt = E + np.cross(drdt,B)
         
        return np.concatenate((drdt,dvdt))
        
    # Starting values
    r0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([0.1, 0.1, 0.1])    
    state = np.concatenate((r0,v0)) #start state
    
    t = np.linspace(0,5,123) # time array 
    
    state = odeint(step, state, t, args=(E,B))
    
    x,y,z = state[:,0], state[:,1], state[:,2]
    
    fig = plt.figure(figsize = (15,10))
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot(x,y,z)
    ax.set_xlabel("X-coord")
    ax.set_ylabel("Y-coord")
    ax.set_zlabel("Z-coord")
    ax.set_title("Particle lifes trajectory")
    plt.show()
    
    print("Particle speed at time t=5: {}".format(state[-1,3:]))
    
    fig2 = plt.figure(figsize=(15,10))
    ax = fig2.add_subplot(111, projection = '3d')
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([0, 0.5])
    ax.set_zlim([0, 0.2])
    ax.set_title("Adventures of particle")
    ax.set_xlabel("X-coord")
    ax.set_ylabel("Y-coord")
    ax.set_zlabel("Z-coord")
    kuva = ax.scatter(state[0,0],state[0,1], state[0,2], marker = 'o', facecolor = 'blue')
    
    def update_scatter(i):
        kuva = ax.scatter(state[i,0], state[i,1], state[i,2], marker = 'o', facecolor = 'blue')
                
    anime = animation.FuncAnimation(fig2, update_scatter, frames = len(t), interval = 20)
    anime.save("particle_world.mp4")


def main():
    
    particle_world()

if __name__=="__main__":
    main()
