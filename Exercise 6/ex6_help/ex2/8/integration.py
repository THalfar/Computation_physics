""" 
-------- EXERCISE 2 - problem 2 ------------
----- FYS-4096 - Computational Physics -----

integration.py contains numerical integration of some
predetermined functions. The answers our own functions give
are tested against values gained from reputable calculation software
such as MatLab and WolframAlpha
"""

import num_calculus as nc
import numpy as np

def integrate_a():
    """
    integrate three functions:
    fun1 from 0 to inf
    fun2 from 0 to 1
    fun3 from 0 to 5
    functions defined as nested functions within this function
    """
    
    def fun1(r): return r**2*np.exp(-2*r)
    def fun2(x): return np.sin(x)/x
    def fun3(x): return np.exp(np.sin(x**3))
    
    print("---------------Integration a----------------")
    print("numerical integration is done with simpson method")
    print()
    
    # exponential function is very close to zero at 1000
    print("integral r^2*exp(-2r)dr from 0 to inf numerically:")
    int1 = nc.num_simpson(fun1,0,1000, 100000)
    print(int1)
    print()
    print("solution to interal solved with MatLab: 0.25")
    print("numerical solution was from 0 to 1000 and with 100000 intervals")
    print("***difference is {}***".format(abs(0.25-int1)))
    print()
    
    # we do not want to include x=0 because it gives 0/0
    print("---------------------------------------------")
    print("integral sin(x)/x dx from 0 to 1 numerically:")
    int2 = nc.num_simpson(fun2,1e-9,1, 10000)
    print(int2)
    print()
    print("solution to interal solved with MatLab: 0.946083070367183")
    print("numerical solution was from 1e-9 to 1 and with 10000 intervals")
    print("***difference is {}***".format(abs(0.946083070367183-int2)))
    print()
    
    
    print("---------------------------------------------")
    print("integral exp(sin(x^3)) dx from 0 to 5 numerically:")
    int3=nc.num_simpson(fun3,0,5, 10000)
    print(int3)
    print()
    print("solution to interal solved with MatLab: 6.647272079953789")
    print("numerical solution was from 0 to 5 and with 10000 intervals")
    print("***difference is {}***".format(abs(6.647272079953789-int3)))
    print()
    
def integrate_b():
    """
    integrate 2d function:
    f(x,y) = x exp(-sqrt(x^2+y^2)) 
    x from 0 to 2
    y from -2 to 2
    function defined as nested function within this function
    """
    
    def fun(x,y): return x*np.exp(-np.sqrt(x**2+y**2))
    def fun_1d(x): return lambda y: fun(x,y)
    
    print("---------------Integration b----------------")
    print("numerical integration is done with trapezoid method")
    print()
    
    print("integral x*exp(-sqrt(x^2+y^2)) dxdy")
    print("x = [0,2], y = [-2,2] numerically:")
    intb = 0
    
    # trapezoidal integral at certain point in x with trapezoidal
    # integral over x
    
    # optimization such that the integral at i needs to be done once
    
    first = nc.num_trapezoid(fun_1d(0),-2,2,2000)*2/2000
    for i in range(0,1999):
    
        second = nc.num_trapezoid(fun_1d((i+1)/1000),-2,2,2000)*2/2000
        
        intb += 0.5 * (first + second)
        first = second
        
    
    print(intb)
    print()
    print("solution to interal solved with WolframAlpha: 1.57347")
    print("numerical solution was with grid x = [0,2] and y = [-2,2]")
    print("with 2000*2000 grid")
    print("***difference is {}***".format(abs(1.57347-intb)))
    print()
    
def integrate_c():
    """
    :param: ra  some definite point xyz
    :param: rb  some definite point xyz
    :param: r   some point xyz that integrate over
    
    integrate function |psi(r-ra)|^2/||r-rb|| over all space
    psi(r) = exp(-r)/sqrt(pi)
    r=||r||=sqrt(x^2+y^2+z^2)
    R = ||ra-rb||
    we will integrate with monte-carlo method
    """
    
    def fun(r): return np.exp(-np.sqrt(r[0]**2+r[1]**2+r[2]**2)) / np.sqrt(np.pi)
    def real(r): return (1-(1+R)*np.exp(-2*R))/R
    
    print("---------------Integration c----------------")
    print("integrates the function |psi(r-ra)|^2/||r-rb|| over all space")
    print("presumably f = 0 when |r-ra|>10, psi(r) = exp(-r)/sqrt(pi)")
    print("r=||r||=sqrt(x^2+y^2+z^2), R = ||ra-rb||")
    print("we will integrate with monte-carlo method five times")
    
    integral = np.zeros(5)
    for k in range(5):
        print("------------{}----------".format(k+1))
        
        first_ra = 20*np.random.rand(3,1) - 10
        first_rb = 20*np.random.rand(3,1) - 10
        R = np.sqrt((first_ra[0] - first_rb[0])**2 +
                    (first_ra[1] - first_rb[1])**2 +
                    (first_ra[2] - first_rb[2])**2)
        
        values = np.zeros((200,))
        V = 20**3
        
        for block in range(200):
            first_r  = 20*np.random.rand(3,2000) - 10
            for i in range(2000):
                vec1 = np.array([first_r[0,i] - first_ra[0],
                                 first_r[1,i] - first_ra[1],
                                 first_r[2,i] - first_ra[2]])
                                 
                vec2 = np.array([first_r[0,i] - first_rb[0],
                                 first_r[1,i] - first_rb[1],
                                 first_r[2,i] - first_rb[2]])
                
                values[block] += abs(fun(vec1)**2)/(np.sqrt(vec2[0]**2+vec2[1]**2+vec2[2]**2))/2000
            
        
        integral[k] = V*np.mean(values)
        
        print("integral value: {}".format(integral[k]))
        
        print("real value is {}".format(real(R)))



def main():
    
    integrate_a()
    integrate_b()
    integrate_c()
    
if __name__=="__main__":
    main()
