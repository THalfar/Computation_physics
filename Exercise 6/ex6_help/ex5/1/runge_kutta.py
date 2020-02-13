"""
Problems 1 and 2 from exercise 5.
Computational physics

Juha Teuho
"""
import numpy as np
from matplotlib.pyplot import *
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

def runge_kutta4(x,t,dt,func,**kwargs):
    """
    Fourth order Runge-Kutta for solving ODEs
    dx/dt = f(x,t)
    
    x = state vector at time t
    t = time
    dt = time step
    
    func = function on the right-hand side,
    i.e., dx/dt = func(x,t;params)
    
    kwargs = possible parameters for the function
             given as args=(a,b,c,...)
    
    Need to complete the routine below
    where it reads '...' !!!!!!!
    
    See FYS-4096 lecture notes.
    """
    F1 = F2 = F3 = F4 = 0.0*x
    if ('args' in kwargs):
        args = kwargs['args']
        F1 = func(x,t,*args)
        F2 = func(x+dt/2*F1,t+dt/2,*args)
        F3 = func(x+dt/2*F2,t+dt/2,*args)
        F4 = func(x+dt*F3,t+dt,*args)
    else:
        F1 = func(x,t)
        F2 = func(x+dt/2*F1,t+dt/2)
        F3 = func(x+dt/2*F2,t+dt/2)
        F4 = func(x+dt*F3,t+dt)

    return x+dt/6*(F1+2*F2+2*F3+F4), t+dt

def pend(y, t, b, c):
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return np.array(dydt)

def pend_ivp(t,y):
    b = 0.25
    c = 5.0
    return pend(y,t,b,c)

def odeint_test(ax):
    b = 0.25
    c = 5.0
    y0 = [np.pi - 0.1, 0.0]
    t = np.linspace(0, 10, 101)
    sol = odeint(pend, y0, t, args=(b, c))
    ax.plot(t, sol[:, 0], 'b', label='theta(t)')
    ax.plot(t, sol[:, 1], 'g', label='omega(t)')
    ax.legend(loc='best')
    ax.set_xlabel('t')
    ax.grid()

def runge_kutta_test(ax):
    b = 0.25
    c = 5.0    
    y0 = [np.pi - 0.1, 0.0]
    t = np.linspace(0, 10, 101)
    dt = t[1]-t[0]
    sol=[]
    x=1.0*np.array(y0)
    for i in range(len(t)):
        sol.append(x)
        x, tp = runge_kutta4(x,t[i],dt,pend,args=(b,c))
    sol=np.array(sol)
    ax.plot(t, sol[:, 0], 'b', label='theta(t)')
    ax.plot(t, sol[:, 1], 'g', label='omega(t)')
    ax.legend(loc='best')
    ax.set_xlabel('t')
    ax.grid()

def solve_ivp_test(ax):
    b = 0.25
    c = 5.0
    y0 = [np.pi - 0.1, 0.0]
    t = np.linspace(0, 10, 101)
    sol = solve_ivp(pend_ivp, (0,10), y0,t_eval=t)
    ax.plot(sol.t, sol.y[0,:], 'b', label='theta(t)')
    ax.plot(sol.t, sol.y[1,:], 'g', label='omega(t)')
    ax.legend(loc='best')
    ax.set_xlabel('t')
    ax.grid()

def func(t, x,q=1,m=1):
    """
    Returns the Lorentz force components and velocities
    
    Args:
        t (float): time
        x (np.ndarray): velocity components and position components
        q (int, optional): elemental charge
        m (int, optional): mass
    
    Returns:
        np.ndarray: Force components and velocity components
    """
    E = np.array([0.05, 0, 0])
    B = np.array([0,4.,0])
    v = np.array([x[0],x[1],x[2]])
    F = q/m*(E+np.cross(v,B))
    return np.array([F[0],F[1],F[2],v[0],v[1],v[2]])

def solveLorentz(initial, tmin, tmax, tstep=0.1):
    """
    Solve trajectory of charged particle under the influence of Lorentz force.
    
    Args:
        initial (np.ndarray): Initial conditions [v_x, v_y, v_z, r_x, r_y, r_z], 
        					  where v is velocity and r is position
        tmin (float): Initial time
        tmax (float): Final time
        tstep (float, optional): Time step
    """

    # Solve the trajectory using solve_ivp from scipy.integrate
    t = np.linspace(tmin,tmax,round((tmax-tmin)/tstep+1))
    sol = solve_ivp(func,(tmin,tmax),initial,t_eval=t)

    return sol


def main():
	# Solve and plot ode using three different methods: scipy ivp, scipy odeint
	# and 4th order Runge-Kutta.
    fig=figure()
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_subplot(133)
    solve_ivp_test(ax1)
    ax1.set_title('solve_ivp')	
    odeint_test(ax2)
    ax2.set_title('odeint')
    runge_kutta_test(ax3)
    ax3.set_title('own Runge-Kutta 4')

    # Plot the trajectory of charged particle under the influence of Lorentz force.
    tmin = 0 # initial time
    tmax = 5 # final time
    tstep = 0.05 # time step
    initial = [0.1,0.1,0.1,0,0,0] # initial conditions: vx,vy,vz,rx,ry,rz

    # calculate the trajectory
    trajectory = solveLorentz(initial,tmin,tmax,tstep).y
    xpoints = trajectory[3,:]
    ypoints = trajectory[4,:]
    zpoints = trajectory[5,:]

    # Plot trajectory
    fig = figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xpoints,ypoints,zpoints)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Trajectory of charged particle')
    show()

if __name__=="__main__":
    main()
