import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from runge_kutta import runge_kutta4


def func(x,t,E,B):
    """
    Calculates F/m, where F=q(E+v x B).
    Inputs : E = qE/m, 
           : B = qB/m,
           : x = [vx,vy,vz,x,y,z]
    Outputs: xx=[dvx/dt,dvy/dt,dvz/dt,vx,vy,vz]

    Can be used in modeling the motion of a charged 
    particle in the influence of both electric and 
    magnetic field.
    """
    xx=0.0*x
    xx[0:3]=E+np.cross(x[0:3],B)
    xx[3:]=x[0:3]
    return xx

def main():
    
    E = np.array([0.05,0,0])
    B = np.array([0,4.,0])
    t = np.linspace(0, 5, 101)
    dt = t[1]-t[0]
    x0 = np.array([0.1,0.1,0.1,0.,0.,0.])
    sol=[]
    x=1.0*np.array(x0)
    for i in range(len(t)):
        sol.append(x)
        x, tp = runge_kutta4(x,t[i],dt,func,args=(E,B))
    print(sol[-1])
    sol=np.array(sol)
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol[:, 3],sol[:, 4],sol[:, 5], 'b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.grid()
    show()

if __name__=="__main__":
    main()
