""" Exercise1 Problem 3: 
(a) numerically estimate first derivative of single argument Python function. 
(b) repeat (a) for second derivative
(c) Riemann, trapezoidal, and Simpson integration functions for case where integrand function values are given on a uniform grid.
(d) Monte Carlo integration
"""

# import needed packages, e.g., import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def first_derivative( function, x, dx ):
# Use Equation (5) from Week 1 lecture notes to estimate first derivative of given function
    dfdx = (function(x + dx) - function(x - dx))/(2*dx)
    return dfdx

def test_first_derivative():
# Compare numerical estimate procedure with known function to verify estimate
    x = 1.2
    dx = 0.0001
    difference = np.abs(first_derivative(fun,x,dx)-fun_der(x))
    if (difference<0.0005):
        print('First derivative is OK!')
    else:
        print('First derivative is NOT ok!')

def fun(x): 
# Test function is 3x^2
    return 3*x**2

def fun_der(x):
# Derivative of test function is 6x
    return 6*x

def second_derivative( function, x, dx ):
# Use Equation (7) from Week 1 lecture notes to estimate second derivative of given function
    dfdx2 = (function(x + dx) + function(x - dx) - 2*function(x))/(dx**2)
    return dfdx2

def test_second_derivative():
# Compare numerical estimate procedure with known function to verify estimate
    x = 1.2
    dx = 0.0001
    difference = np.abs(second_derivative(fun,x,dx)-fun_der_2(x))
    if (difference<0.0005):
        print('Second derivative is OK!')
    else:
        print('Second derivative is NOT ok!')

def fun_der_2(x):
# Second derivative of test function is 6
    return 6

def riemann_sum( f, x ):
# Use Equation (8) from Week 1 lecture notes to estimate integral of given function
    rie = 0
    for n in range(len(x) - 1):
        rie = rie + f[n]*(x[n+1]-x[n])
    return rie

def test_riemann_sum():
# Compare numerical estimate procedure with known function to verify estimate
    x = np.linspace(0,np.pi/2,100)
    f = np.sin(x)
    I = riemann_sum(f,x)
    print("Riemann sum = ", I)
    difference = np.abs(I - fun_rie(x))
    if (difference < 0.02):
        print("Riemann sum is OK!")
    else:
        print("Riemann sum is NOT ok!")

def fun_rie(x):
# Test function is sin(x) integrated from 0 to pi/2
    return 1

def trapezoid( f, x ):
# Use Equation (9) from Week 1 lecture notes to estimate integral of given function
    trap = 0
    for n in range(len(x) - 1):
        trap = trap + 0.5*(f[n] + f[n + 1])*(x[n+1]-x[n])
    return trap

def test_trapezoid():
# Compare numerical estimate procedure with known function to verify estimate
    x = np.linspace(0,np.pi/2,100)
    f = np.sin(x)
    I = trapezoid(f,x)
    print("Trapezoid = ", I)
    difference = np.abs(I - fun_trap(x))
    if (difference < 0.02):
        print("Trapezoid rule test is OK!")
    else:
        print("Trapezoid rule test is NOT ok!")

def fun_trap(x):
# Test function is sin(x) integrated from 0 to pi/2
    return 1

def simpson_even( f, x ):
# Use Equation (10) from Week 1 lecture notes to estimate integral of given function
# with even number of intervals
    simp_even = 0
    for n in range(int(len(x)/2) - 1):
        simp_even = simp_even + ((x[n+1]-x[n])/3)*(f[2*n] + 4*f[2*n + 1] + f[2*n + 2])
    return simp_even

def test_simpson_even():
# Compare numerical estimate procedure with known function to verify estimate
    x = np.linspace(0,np.pi/2,100)
    f = np.sin(x)
    I = simpson_even(f,x)
    print("Simpson, even number intervals = ", I)
    difference = np.abs(I - fun_simp_even(x))
    if (difference < 0.02):
        print("Simpson rule test with even number of intervals is OK!")
    else:
        print("Simpson rule test with even number of intervals is NOT ok!")

def fun_simp_even(x):
# Test function is sin(x) integrated from 0 to pi/2
    return 1

def simpson_odd( f, x ):
# Use Equations (10) & (11) from Week 1 lecture notes to estimate integral of given function
# with odd number of intervals
    delta_I = ((x[2]-x[1])/12)*(0 - f[len(x) - 3] + 8*f[len(x) - 2] + 5*f[len(x) - 1])
    simp_even = 0
    for n in range(int((len(x)-1)/2) - 1):
        simp_even = simp_even + ((x[n+1]-x[n])/3)*(f[2*n] + 4*f[2*n + 1] + f[2*n + 2])
    simp_odd = simp_even + delta_I
    return simp_odd

def test_simpson_odd():
# Compare numerical estimate procedure with known function to verify estimate
    x = np.linspace(0,np.pi/2,101)
    f = np.sin(x)
    I = simpson_odd(f,x)
    print("Simpson, odd number intervals = ", I)
    difference = np.abs(I - fun_simp_odd(x))
    if (difference < 0.02):
        print("Simpson rule test with odd number of intervals is OK!")
    else:
        print("Simpson rule test with odd number of intervals is NOT ok!")

def fun_simp_odd(x):
# Test function is sin(x) integrated from 0 to pi/2
    return 1

def monte_carlo_integration(fun,xmin,xmax,blocks,iters):
# Use instructions from exercise1.pdf to estimate integral of given function
    block_values=np.zeros((blocks,))
    L=xmax-xmin
    for block in range(blocks):
        for i in range(iters):
            x = xmin+np.random.rand()*L
            block_values[block]+=fun(x)
        block_values[block]/=iters
    I = L*np.mean(block_values)
    dI = L*np.std(block_values)/np.sqrt(blocks)
    return I,dI

def func(x):
# Test function for Monte Carlo integration
    return np.sin(x)

def test_MC(I, dI):
# Compare numerical estimate procedure with known function to verify estimate
    difference = np.abs(I - fun_MC())
    if (difference < 0.02):
        print("Monte Carlo estimate is OK!")
    else:
        print("Monte Carlo estimate is NOT ok!")

def fun_MC():
# Test function is sin(x) integrated from 0 to pi/2
    return 1

def main():
# Tests of Problem 3, parts (a) through (d): first and second derivatives; Riemann, Trapezoid, Simpson; Monte Carlo
    test_first_derivative()
    test_second_derivative()
    test_riemann_sum()
    test_trapezoid()
    test_simpson_even()
    test_simpson_odd()
    I,dI=monte_carlo_integration(func,0.,np.pi/2,10,100)
    print("Monte Carlo = ", I,'+/-',2*dI)
    test_MC(I, dI)

if __name__=="__main__":
    main()
