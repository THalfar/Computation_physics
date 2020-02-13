"""
Fourth order Runge-Kutta

Related to FYS-4096 Computational Physics
exercise 5 assignments.

Modified by Roman Goncharov on February 2020
"""
import numpy as np
from matplotlib.pyplot import *

"""two solving methods imported here"""
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

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

    For defined F's see Week 4 FYS-4096 Computational Physics 
    lecture slides.
    """
    F1 = F2 = F3 = F4 = 0.0*x
    if ('args' in kwargs):
        args = kwargs['args']
        F1 = func(x,t,*args)
        F2 = func(x+dt * F1 / 2,t + dt / 2,*args)
        F3 = func(x+dt * F2 / 2,t + dt / 2,*args)
        F4 = func(x+dt * F3,t + dt,*args)
    else:
        F1 = func(x,t)
        F2 = func(x+dt * F1 / 2,t + dt / 2)
        F3 = func(x+dt * F2 / 2,t + dt / 2)
        F4 = func(x+dt * F3,t + dt)

    return x+dt * (F1 + 2 * F2 + 2 * F3 + F4) / 6, t+dt

def pend(y, t, b, c):
    """
    Pendulum function for test perfoming 
    y = solutions
    t = time variable
    b,c = parameters
    for details see e.g. Wikipedia
    """
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return np.array(dydt)

def pend_ivp(t,y):
    """
    Fixing parameters for given function
    """
    b = 0.25
    c = 5.0
    return pend(y,t,b,c)

def odeint_test(ax):
    """Test perfoming for odeint"""
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
    """Test perfoming for Runge-Kutta method"""
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
    """Test perfoming for solve_ivp"""
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

def error_check():
    """Error checker, gives the absolute difference between built-in and handmade methods"""
    b = 0.25
    c = 5.0
    y0 = [np.pi - 0.1, 0.0]
    t = np.linspace(0, 10, 101)
    sol1 = odeint(pend, y0, t, args=(b, c))
   
    dt = t[1]-t[0]
    sol2=[]
    x=1.0*np.array(y0)
    for i in range(len(t)):
        sol2.append(x)
        x, tp = runge_kutta4(x,t[i],dt,pend,args=(b,c))
    sol2=np.array(sol2)    
    print('Comparison of R-K method and SciPy method. Error: ',abs(np.amax(sol1)-np.amax(sol2)))
    
def main():
    """
    All the necessary tests and plotting  perfomed in main()
    """
    error_check()
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
    show()
    


if __name__=="__main__":
    main()
