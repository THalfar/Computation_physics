"""
calculating some numerical integrals for exercise 2 problem 2
"""
from scipy.integrate import simps
from scipy.integrate import dblquad
from scipy.integrate import tplquad
import numpy as np

# functions for integration in a)
def fun1(x):
    return x**2*np.exp(-2*x)

def fun2(x):
    return np.sin(x)/x

def fun3(x):
    return np.exp(np.sin(x**3))

def a_integrals():
    print("a)")
    # function is very close to 0 and fun->0 when x->inf so using
    # lots of points with small upperlimit
    x=np.linspace(0, 50, 1000000)
    print("this should be 0.25")
    print(simps(fun1(x), x))

    # not defined at 0 so using something very close to 0 as starting point
    x=np.linspace(10e-10, 1, 1000000)
    print("this should be about 0,946083")
    print(simps(fun2(x), x))

    x=np.linspace(0, 5, 1000000)
    print("this should be about 6.647272")
    print(simps(fun3(x), x))

# function for b)
def fun4(x,y):
    return x*np.exp(-np.sqrt(x**2+y**2))

def b_integral():
    # calculating with https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.dblquad.html
    print("b)")
    value, error =dblquad(fun4,0,2,-2,2)
    print("value: ",value, "error estimate: ", error)

def psi(r):
    # function for integral in c)
    return np.exp(-r)/np.sqrt(np.pi)

def r(x,y,z):
    return np.sqrt(x**2+y**2+z**2)

def fun5(x,y,z):
    return abs(psi(r(x,y,z)-r(1,0,0)))**2/abs(psi(r(x,y,z)-r(0,0,0)))

def c_integral():
    print("c)")
    value, error =tplquad(fun5,-10e5,10e5,-10e5,10e5,-10e5,10e5)
    print("value: ",value, "error estimate: ", error)

def main():
    a_integrals()
    b_integral()
    # doesn't work runtime warning i couldn't figure out
    # c_integral()

if __name__ == "__main__":
    main()
