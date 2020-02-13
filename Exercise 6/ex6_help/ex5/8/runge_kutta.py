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
        # Runge kutta F formulas from lecture slides

        F1 = func(x,t,*args)
        F2 = func(x + dt/2*F1, t + dt/2, *args)
        F3 = func(x + dt/2*F2, t + dt/2, *args)
        F4 = func(x + dt*F3, t + dt, *args)
    else:
        F1 = func(x,t)
        F2 = func(x + dt / 2 * F1, t + dt / 2)
        F3 = func(x + dt / 2 * F2, t + dt / 2)
        F4 = func(x + dt * F3, t + dt)

    new_x = x + dt/6*(F1 + 2*F2 + 2*F3 + F2)
    # return updated vector and time
    return new_x, t+dt

def pend(y, t, b, c):
    # all state variables are contained in y, and function returns the derivatives of each state variable
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return np.array(dydt)

def pend_ivp(t,y):
    b = 0.25
    c = 5.0
    return pend(y,t,b,c)

def odeint_test(ax, step_n=101):
    # simple function for solving and plotting the pendulum motion with odeint method from scipy
    b = 0.25
    c = 5.0
    y0 = [np.pi - 0.1, 0.0]
    t = np.linspace(0, 10, step_n)
    sol = odeint(pend, y0, t, args=(b, c))
    ax.plot(t, sol[:, 0], 'b', label='theta(t)')
    ax.plot(t, sol[:, 1], 'g', label='omega(t)')
    ax.legend(loc='best')
    ax.set_xlabel('t')
    ax.grid()

def runge_kutta_test(ax, step_n=101):
    # simple function for solving and plotting the pendulum motion with runge_kutta4 function
    b = 0.25
    c = 5.0    
    y0 = [np.pi - 0.1, 0.0]  # starting position
    t = np.linspace(0, 10, step_n)  # timesteps that are evaluated
    dt = t[1]-t[0]
    sol=[]
    x=1.0*np.array(y0)

    # solve RK one timestep at a time, and add the solutions to sol list
    for i in range(len(t)):
        sol.append(x)
        x, tp = runge_kutta4(x,t[i],dt,pend,args=(b,c))
    sol=np.array(sol)
    ax.plot(t, sol[:, 0], 'b', label='theta(t)')
    ax.plot(t, sol[:, 1], 'g', label='omega(t)')
    ax.legend(loc='best')
    ax.set_xlabel('t')
    ax.grid()

def solve_ivp_test(ax, step_n=101):
    # simple function for solving and plotting the pendulum motion with solve_ivp method from scipy

    b = 0.25
    c = 5.0
    y0 = [np.pi - 0.1, 0.0]
    t = np.linspace(0, 10, step_n)
    sol = solve_ivp(pend_ivp, (0,10), y0,t_eval=t)
    ax.plot(sol.t, sol.y[0,:], 'b', label='theta(t)')
    ax.plot(sol.t, sol.y[1,:], 'g', label='omega(t)')
    ax.legend(loc='best')
    ax.set_xlabel('t')
    ax.grid()


def compare_RK_with_scipy(step_n, scipy_method='odeint'):
    """
        Function compares RK with the scipy methods (either odeint or solve_ivp)
        creates a figure with the average difference between method solutions.
    :param step_n: how many timesteps are used
    :param scipy_method: either 'odeint' (default) or 'solve_ivp'
    """
    b = 0.25
    c = 5.0
    y0 = [np.pi - 0.1, 0.0]
    t = np.linspace(0, 10, step_n)
    dt = t[1] - t[0]

    if scipy_method == 'odeint':
        sol_scipy = odeint(pend, y0, t, args=(b, c))
    elif scipy_method == 'solve_ivp':
        sol_scipy = solve_ivp(pend_ivp, (t[0], t[-1]), y0, t_eval=t)
    else:
        print("Error in function compare_RK_with_scipy: Choose either odeint or solve_ivp method!")
        return

    sol_rk = []
    x = 1.0 * np.array(y0)
    for i in range(len(t)):
        sol_rk.append(x)
        x, tp = runge_kutta4(x, t[i], dt, pend, args=(b, c))
    sol_rk = np.array(sol_rk)

    if scipy_method == 'odeint':
        theta_diff = sol_rk[:, 0] - sol_scipy[:, 0]
        omega_diff = sol_rk[:, 1] - sol_scipy[:, 1]
    elif scipy_method == 'solve_ivp':
        theta_diff = sol_rk[:, 0] - sol_scipy.y[0,:]
        omega_diff = sol_rk[:, 1] - sol_scipy.y[1,:]

    figure()
    plot(t, theta_diff, 'b', label='theta(t), R-K minus ' + scipy_method)
    plot(t, omega_diff, 'g', label='omega(t), R-K minus ' + scipy_method)
    legend(loc='best')
    xlabel('t')
    ylabel('diff')
    title('Difference between runge-kutta and ' + scipy_method + ' solutions\n'
          'positive diff = runge-kutta value is higher')
    grid()


def main():

    fig=figure()
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_subplot(133)

    step_n = 300
    solve_ivp_test(ax1, step_n)
    ax1.set_title('solve_ivp')
    odeint_test(ax2, step_n)
    ax2.set_title('odeint')
    runge_kutta_test(ax3, step_n)
    ax3.set_title('own Runge-Kutta 4')

    compare_RK_with_scipy(step_n, 'odeint')
    compare_RK_with_scipy(step_n, 'solve_ivp')
    show()


if __name__=="__main__":
    main()
