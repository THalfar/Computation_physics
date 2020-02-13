import numpy as np
from matplotlib.pyplot import *
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

    Need to complete the routine below
    where it reads '...' !!!!!!!
    
    See FYS-4096 lecture notes.
    """
    F1 = F2 = F3 = F4 = 0.0*x
    # functions from week 5 lecture notes formulas 49-52
    # used for function with parameters and also without parameters
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
    # returning fuction 48 from week 5 lecture notes, and also time after step
    return x+dt/6*(F1+2*F2+2*F3+F4), t+dt

def pend(y, t, b, c):
    """
    diff equation for testing
    :param y: starting y values values
    :param t: time values
    :param b: constant
    :param c: constant
    :return: diff equation array
    """
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return np.array(dydt)

def pend_ivp(t,y):
    b = 0.25
    c = 5.0
    return pend(y,t,b,c)

def odeint_test(ax):
    """
    solve example with scipys odeint and plot the solution
    :param ax: subplot
    """
    # initializing parameters
    b = 0.25
    c = 5.0
    y0 = [np.pi - 0.1, 0.0]
    t = np.linspace(0, 10, 101)
    # solving with odeint
    sol = odeint(pend, y0, t, args=(b, c))
    # printing max absolute diff
    print("odeint:       ",np.amax(abs(sol)))
    # plotting
    ax.plot(t, sol[:, 0], 'b', label='theta(t)')
    ax.plot(t, sol[:, 1], 'g', label='omega(t)')
    ax.legend(loc='best')
    ax.set_xlabel('t')
    ax.grid()

def runge_kutta_test(ax):
    """
    solve example with own runge_kutta4 function and plot the solution
    :param ax: subplot
    """
    # initializing parameters
    b = 0.25
    c = 5.0    
    y0 = [np.pi - 0.1, 0.0]
    t = np.linspace(0, 10, 101)
    dt = t[1]-t[0]
    sol=[]
    x=1.0*np.array(y0)
    # using runge_kutta function
    for i in range(len(t)):
        sol.append(x)
        x, tp = runge_kutta4(x,t[i],dt,pend,args=(b,c))
    sol=np.array(sol)
    # printing max absolute diff
    print("runge-kutta4: ", np.amax(abs(sol)))
    # plotting result
    ax.plot(t, sol[:, 0], 'b', label='theta(t)')
    ax.plot(t, sol[:, 1], 'g', label='omega(t)')
    ax.legend(loc='best')
    ax.set_xlabel('t')
    ax.grid()

def solve_ivp_test(ax):
    """
    solve example with scipys solve_ivp initial value problem solver and plotting the solution
    :param ax: subplot
    """
    # initializing parameters
    b = 0.25
    c = 5.0
    y0 = [np.pi - 0.1, 0.0]
    t = np.linspace(0, 10, 101)
    # solving diff equation
    sol = solve_ivp(pend_ivp, (0,10), y0,t_eval=t)
    # printing max absolute diff
    print("solve ivp:    ", np.amax(abs(sol.y)))
    # plotting results
    ax.plot(sol.t, sol.y[0,:], 'b', label='theta(t)')
    ax.plot(sol.t, sol.y[1,:], 'g', label='omega(t)')
    ax.legend(loc='best')
    ax.set_xlabel('t')
    ax.grid()

def main():

    # making new figure with 3 subplots
    fig=figure()
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_subplot(133)
    # solving example function with all 3 methods
    solve_ivp_test(ax1)
    ax1.set_title('solve_ivp')
    odeint_test(ax2)
    ax2.set_title('odeint')
    runge_kutta_test(ax3)
    ax3.set_title('own Runge-Kutta 4')
    show()
    


if __name__=="__main__":
    main()
