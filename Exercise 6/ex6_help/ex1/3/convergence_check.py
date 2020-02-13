""" Exercise1 Problem 4: 
Demonstrating the convergence properties of the functions implemented in Problem 3 (num_calculus.py)
"""

# import needed packages, e.g., import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def first_derivative( function, x, dx ):
# Use Equation (5) from Week 1 lecture notes to estimate first derivative of given function
    dfdx = (function(x + dx) - function(x - dx))/(2*dx)
    return dfdx

def test_first_derivative(dx):
# Compare numerical estimate procedure with known function to verify estimate
    x = 1.2
#    dx = 0.0001
    difference = np.abs(first_derivative(fun,x,dx)-fun_der(x))
    return difference

def fun(x): 
# Test function is sin(x)
    return np.sin(x)

def fun_der(x):
# Derivative of test function is cos(x)
    return np.cos(x)

def second_derivative( function, x, dx ):
# Use Equation (7) from Week 1 lecture notes to estimate second derivative of given function
    dfdx2 = (function(x + dx) + function(x - dx) - 2*function(x))/(dx**2)
    return dfdx2

def test_second_derivative(dx):
# Compare numerical estimate procedure with known function to verify estimate
    x = 1.2
#    dx = 0.0001
    difference = np.abs(second_derivative(fun,x,dx)-fun_der_2(x))
    return difference

def trapezoid( f, x, dx):
# Use Equation (9) from Week 1 lecture notes to estimate integral of given function
    trap = 0
    for n in range(len(x) - 1):
        trap = trap + 0.5*(f[n] + f[n + 1])*dx
    return trap

def test_trapezoid(dx):
# Compare numerical estimate procedure with known function 
    x = np.linspace(0,np.pi/2,int(np.pi/(2*dx)))
    f = np.sin(x)
    I = trapezoid(f,x,dx)
    difference = np.abs(I - fun_trap(x))
    return difference

def fun_trap(x):
# Test function is sin(x) integrated from 0 to pi/2
    return 1

def fun_der_2(x):
# Second derivative of test function is -sin(x)
    return np.sin(x)*(-1)

def abs_err(val,dx,a):
# Calculating the absolute error for bin sizes given by vector dx
    if a == 1:
        for n in range(len(dx)):
            val[n] = test_first_derivative(dx[n])
    if a == 2:
        for n in range(len(dx)):
            val[n] = test_second_derivative(dx[n])
    if a == 3:
        for n in range(len(dx)):
            val[n] = test_trapezoid(dx[n])
    my_plot(dx,val,a)
    return val

def my_plot(x,f,a):
# Plotting absolute error as function of bin sizes given by vector dx
    fig = plt.figure()
    plt.plot(x,f)#,label=
    plt.xlabel(r'$dx$')
    if a == 1:
        plt.ylabel(r"First Derivative Absolute Error")
        fig.savefig('AbsErrFirstDer.pdf',dpi=200)
    if a == 2:
        plt.ylabel(r'Second Derivative Absolute Error')
        fig.savefig('AbsErrSecDer.pdf',dpi=200)
    if a == 3:
        plt.ylabel(r'Trapezoid Rule Absolute Error')
        fig.savefig('AbsErrTrapRule.pdf',dpi=200)

def main():
# Absolute error as function of bin size dx of first and second derivatives and Trapezoid rule estimate vs. known values
    dx = np.linspace(1e-4,1,int(1e4))
    val = np.zeros(len(dx))
    first_der_err = abs_err(val,dx,1)
    second_der_err = abs_err(val,dx,2)
    trap_err = abs_err(val,dx,3)

if __name__=="__main__":
    main()

fig = plt.figure()
# - or, e.g., fig = plt.figure(figsize=(width, height))
# - so-called golden ratio: width=height*(sqrt(5.0)+1.0)/2.0
ax = fig.add_subplot(111)

x = np.linspace(0,np.pi/2,100)
f = np.sin(x)
dfdx = np.cos(x)
g = np.exp(-x)
dgdx = np.exp(-x)*(-1)

# plot and add label if legend desired
ax.plot(x,f,label=r'$f(x)=\sin(x)$')
ax.plot(x,g,label=r'$f(x)=\exp(-x)$')

# plot legend
ax.legend(loc=0)

# set axes labels and limits
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'f(x)')
ax.set_xlim(x.min(), x.max())
fig.tight_layout(pad=1)

# save figure as pdf with 200dpi resolution
fig.savefig('testfile.pdf',dpi=200)
plt.show()
