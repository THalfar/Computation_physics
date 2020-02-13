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
    if ('args' in kwargs):
        args = kwargs['args']
        F1 = func(x,t,*args)
        F2 = func(x+0.5*dt*F1,t+0.5*dt,*args)
        F3 = func(x+0.5*dt*F2,t+0.5*dt,*args)
        F4 = func(x+dt*F3,t+dt,*args)
    else:
        F1 = func(x,t)
        F2 = func(x+0.5*dt*F1,t+0.5*dt)
        F3 = func(x+0.5*dt*F2,t+0.5*dt)
        F4 = func(x+dt*F3,t+dt)

    return x+(dt/6)*(F1+2*F2+2*F3+F4), t+dt

def pend(y, t, b, c):
    
    # function which the evolution is solved
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return np.array(dydt)

def pend_ivp(t,y):
    
    # function altered for ivp
    b = 0.25
    c = 5.0
    return pend(y,t,b,c)

def odeint_test(ax):

    # define function
    b = 0.25
    c = 5.0
    y0 = [np.pi - 0.1, 0.0]
    t = np.linspace(0, 10, 101)
    
    # calculate solution
    sol = odeint(pend, y0, t, args=(b, c))
    
    # plot
    ax.plot(t, sol[:, 0], 'b', label='theta(t)')
    ax.plot(t, sol[:, 1], 'g', label='omega(t)')
    ax.legend(loc='best')
    ax.set_xlabel('t')
    ax.grid()
    return sol

def runge_kutta_test(ax):

    # define funciton
    b = 0.25
    c = 5.0    
    y0 = [np.pi - 0.1, 0.0]
    t = np.linspace(0, 10, 101)
    dt = t[1]-t[0]
    
    # calculate solution
    sol=[]
    x=1.0*np.array(y0)
    for i in range(len(t)):
        sol.append(x)
        x, tp = runge_kutta4(x,t[i],dt,pend,args=(b,c))
    sol=np.array(sol)
    
    # plot
    ax.plot(t, sol[:, 0], 'b', label='theta(t)')
    ax.plot(t, sol[:, 1], 'g', label='omega(t)')
    ax.legend(loc='best')
    ax.set_xlabel('t')
    ax.grid()
    return sol

def solve_ivp_test(ax):

    # define the function
    b = 0.25
    c = 5.0
    y0 = [np.pi - 0.1, 0.0]
    t = np.linspace(0, 10, 101)
    
    # calculate the solution
    sol = solve_ivp(pend_ivp, (0,10), y0,t_eval=t)
    
    # plot
    ax.plot(sol.t, sol.y[0,:], 'b', label='theta(t)')
    ax.plot(sol.t, sol.y[1,:], 'g', label='omega(t)')
    ax.legend(loc='best')
    ax.set_xlabel('t')
    ax.grid()
    return sol.y

def main():
    
    # crate figure 
    fig=figure()
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_subplot(133)
    
    # plotting happens inside functions
    sol_ivp = solve_ivp_test(ax1)
    ax1.set_title('solve_ivp')
    sol_ode = odeint_test(ax2)
    ax2.set_title('odeint')
    sol_runge = runge_kutta_test(ax3)
    ax3.set_title('own Runge-Kutta 4')
    show()
    
    # calculate absolute difference of runge-kutta with respect to scipy 
    abs_ode = np.amax(abs(sol_runge-sol_ode))
    abs_ivp = np.amax(abs(sol_runge-np.transpose(sol_ivp)))
    print(f"Maximum absolute difference of 4th order Runge-Kutta \n" \
          f"compared to odeint    function is: {abs_ode} \n" \
          f"compared to solve_ivp function is: {abs_ivp}")
    


if __name__=="__main__":
    main()
