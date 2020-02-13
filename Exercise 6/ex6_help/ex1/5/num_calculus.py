""" 
-------- EXERCISE 1 - problem 3 ------------
----- FYS-4096 - Computational Physics -----

num_calculus.py contains functions for 
numerical calculus e.g. differentiation as in 
numerical derivations for certain order or
numerical integration like Riemann sum 

"""

import numpy as np

def first_derivative( function, x, dx ):
    # returns the frist derivative of function
    
    # function: function to derivate
    # x       : location of derivation
    # dx      : accuracy of the derivative
    
    return num_derivative( function, x, dx, 1)
    
def second_derivative( function, x, dx ):
    # returns the second derivative of function
    
    # function: function to derivate
    # x       : location of derivation
    # dx      : accuracy of the derivative
    
    return num_derivative( function, x, dx, 2)

def num_derivative( function, x, dx, order ):
    # function: function to derivate
    # x       : location of derivation
    # dx      : accuracy of the derivative
    # order   : order of wanted derivative (1 or 2)
    
    # solves linear system with matrix calculus:
    # f(x+dx) ~ f(x) + dx*f'(x) + h^2/2*f"(x)
    # f(x-dx) ~ f(x) - dx*f'(x) + h^2/2*f"(x)
    # as xv = A^-1 * b | xv = [[f'/dx],[f"/dx^2]]
    # and returns derivative of order "order"
    
    A = np.matrix([[0.5,-0.5],
                   [1  , 1  ]])
                   
    b = np.matrix([[function(x+dx)-function(x)],
                   [function(x-dx)-function(x)]])
    
    xv = A*b
    
    return xv.item(order-1)/(dx**order)

def num_riemann( function, a, b, n , right = 0 ):
    # function : function to integrate
    # a        : lower bound of integral
    # b        : higher bound of integral
    # n        : number of intervals > 1
    # right    : boolean for right/left (right = 1/0)
    #            side integral

    # calculates riemann sum as:
    # I = sum(f(x) * dx) = sum(f(x)) * dx  
    # left/right side determined by leaving either 
    # last/first value of x - vector
    # the grid is set as uniform
    
    x = np.linspace(a,b,n+1)

    return sum(function(x[right:n+right]))*(b-a)/(n-1)

def num_trapezoid(function, a, b, n ):
    # function : function to integrate
    # a        : lower bound of integral
    # b        : higher bound of integral
    # n        : number of intervals > 1

    # calculates trapezoidal instegral sum as:
    # I = sum(0.5*(f(x)+f(x+dx)) * dx) 
    #   = 0.5 * sum(f(x)+f(x+dx)) * dx
    # the grid is set as uniform
    
    x = np.linspace(a,b,n+1)
    
    return 0.5*sum(function(x[0:n])+function(x[1:n+1]))*(b-a)/(n-1)
    
def num_simpson(function, a, b, n ):
    # function : function to integrate
    # a        : lower bound of integral
    # b        : higher bound of integral
    # n        : number of intervals > 1, must be even!

    # calculates simpson instegral sum as:
    # I = sum(1/3*(f(2x)+4*f(2x+dx)+f(2x+2dx)) * dx) 
    #   = 1/3 * sum(f(2x)+4*f(2x+dx)+f(2x+2dx)) * dx
    # the grid is set as uniform
    
    x = np.linspace(a,b,n+1)
    
    return 1/3 * sum(function(x[0:n-1:2])+4*function(x[1:n:2])+function(x[2:n+1:2])) * (b-a)/n
    
def monte_carlo_integration(fun ,xmin ,xmax ,blocks ,iters ):
    # fun     : function to integrate
    # xmin    : lower bound of integral
    # xmax    : higher bound of integral
    # blocks  : number of integration blocks
    #           integral will be mean of these
    # iters   : number of iterations in block
    
    # calculates approximate integral with monte carlo method
    # I = (xmax-xmin)/(blocks+iters)*sum(f(rand)) +- std blocks
    
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

def test_first_derivative():
    # tests if functions in this file work properly
    # i.e. if value is within promille of correct
    
    def fun1(x): return 3*x**2
    
    value = first_derivative(fun1, 2, 0.001)
    correct = 12
    
    if abs(value - correct)/correct < 0.001:
        print("first derivative correct for simple polynomials")
        
    else:
        print("first derivative incorrect for simple polynomials")
    
    print("correct value is <{}>, functions value is <{}>".format(correct,value))
    print("difference is {} %".format(abs(value-correct)*100))
    print()
    print("function checked was 3*x^2")
    
    print()
    
    def fun2(x): return np.sin(x)
    
    value = first_derivative(fun2, 2, 0.001)
    correct = -0.4161468365
    
    if abs(value - correct)/correct < 0.001:
        print("first derivative correct for a sine function")
        
    else:
        print("first derivative incorrect for a sine function")
        
    print("correct value is <{}>, functions value is <{}>".format(correct,value))
    print("difference is {} %".format(abs(value-correct)*100))
    print()
    print("function checked was sin(x)")
    print()
    
    return 0

def test_second_derivative():
    # tests if functions in this file work properly
    # i.e. if value is within promille of correct
    
    def fun1(x): return 3*x**2
    
    value = second_derivative(fun1, 2, 0.001)
    correct = 6
    
    if abs(value - correct)/correct < 0.001:
        print("second derivative correct for simple polynomials")
        
    else:
        print("second derivative incorrect for simple polynomials")
    
    print("correct value is <{}>, functions value is <{}>".format(correct,value))
    print("difference is {} %".format(abs(value-correct)*100))
    print()
    print("function checked was 3*x^2")
        
    print()
    
    def fun2(x): return np.sin(x)
    
    value = second_derivative(fun2, 2, 0.001)
    correct = -0.909297427
    
    if abs(value - correct)/correct < 0.001:
        print("second derivative correct for a sine function")
        
    else:
        print("second derivative incorrect for a sine function")
    
    print("correct value is <{}>, functions value is <{}>".format(correct,value))
    print("difference is {} %".format(abs(value-correct)*100))
    print()
    print("function checked was sin(x)")
    print()
    
    
    return 0
    
def test_num_riemann():
    # tests if functions in this file work properly
    # i.e. if value is within promille of correct
    failures = 0
    
    def fun1(x): return 3*x**2
    
    value = num_riemann(fun1, 0, 2, 100000)
    correct = 8
    
    if abs(value - correct)/correct < 0.001:
        print("left side riemann sum correct for simple polynomials")
        
    else:
        print("left side riemann sum incorrect for simple polynomials")
        failures += 1
    
    print("correct value is <{}>, functions value is <{}>".format(correct,value))
    print("difference is {} %".format(abs(value-correct)*100))
    print()
        
    value = num_riemann(fun1, 0, 2, 100000,1)
    
    if abs(value - correct)/correct < 0.001:
        print("right side riemann sum correct for simple polynomials")
        
    else:
        print("right side riemann sum incorrect for simple polynomials")
        failures += 1
    
    print("correct value is <{}>, functions value is <{}>".format(correct,value))
    print("difference is {} %".format(abs(value-correct)*100))
    print()
    print("function checked was 3*x^2")
    print("number of intervals was 100000")
    print()
    
    def fun2(x): return np.sin(x)
    
    value = num_riemann(fun2, 0, 2, 100000)
    correct = 1.4161468365
    
    if abs(value - correct)/correct < 0.001:
        print("left side riemann sum correct for a sine function")
        
    else:
        print("left side riemann sum incorrect for a sine function")
        failures += 1
    
    print("correct value is <{}>, functions value is <{}>".format(correct,value))
    print("difference is {} %".format(abs(value-correct)*100))
    print()
    
    value = num_riemann(fun2, 0, 2, 100000, 1)
    
    if abs(value - correct)/correct < 0.001:
        print("right side riemann sum correct for a sine function")
        
    else:
        print("right side riemann sum incorrect for a sine function")
        failures += 1
    
    print("correct value is <{}>, functions value is <{}>".format(correct,value))
    print("difference is {} %".format(abs(value-correct)*100))
    print()
    print("function checked was sin(x)")
    print("number of intervals was 100000")
    print()
    
    return failures

def test_num_trapezoid():
    # tests if functions in this file work properly
    # i.e. if value is within promille of correct
    failures = 0
    
    def fun1(x): return 3*x**2
    
    value = num_trapezoid(fun1, 0, 2, 100000)
    correct = 8
    
    if abs(value - correct)/correct < 0.001:
        print("trapezoidal integral correct for simple polynomials")
        
    else:
        print("trapezoidal integral incorrect for simple polynomials")
        failures += 1
    
    print("correct value is <{}>, functions value is <{}>".format(correct,value))
    print("difference is {} %".format(abs(value-correct)*100))
    print()
        
    print("function checked was 3*x^2")
    print("number of intervals was 100000")
    print()
    
    def fun2(x): return np.sin(x)
    
    value = num_trapezoid(fun2, 0, 2, 100000)
    correct = 1.4161468365
    
    if abs(value - correct)/correct < 0.001:
        print("trapezoidal integral sum correct for a sine function")
        
    else:
        print("trapezoidal integral sum incorrect for a sine function")
        failures += 1
    
    print("correct value is <{}>, functions value is <{}>".format(correct,value))
    print("difference is {} %".format(abs(value-correct)*100))
    print()
    print("function checked was sin(x)")
    print("number of intervals was 100000")
    print()
    
    return failures
    
def test_num_simpson():
    # tests if functions in this file work properly
    # i.e. if value is within promille of correct
    failures = 0
    
    def fun1(x): return 3*x**2
    
    value = num_simpson(fun1, 0, 2, 100)
    correct = 8
    
    if abs(value - correct)/correct < 0.001:
        print("simpson integral correct for simple polynomials")
        
    else:
        print("simpson integral incorrect for simple polynomials")
        failures += 1
    
    print("correct value is <{}>, functions value is <{}>".format(correct,value))
    print("difference is {} %".format(abs(value-correct)*100))
    print()
        
    print("function checked was 3*x^2")
    print("number of intervals was 100")
    print()
    
    def fun2(x): return np.sin(x)
    
    value = num_simpson(fun2, 0, 2, 100)
    correct = 1.4161468365
    
    if abs(value - correct)/correct < 0.001:
        print("simpson integral sum correct for a sine function")
        
    else:
        print("simpson integral sum incorrect for a sine function")
        failures += 1
    
    print("correct value is <{}>, functions value is <{}>".format(correct,value))
    print("difference is {} %".format(abs(value-correct)*100))
    print()
    print("function checked was sin(x)")
    print("number of intervals was 100")
    print()
    
    return failures
    
def test_monte_carlo():
    # tests if functions in this file work properly
    # i.e. if value is within promille of correct
    
    failures = 0
    
    n = 1000
    
    def fun1(x): return 3*x**2
    
    value = monte_carlo_integration(fun1, 0, 2, n, n)
    correct = 8
    
    if correct <= value[0] + value[1] and correct >= value[0] - value[1]:
        print("monte carlo integral correct for simple polynomials")
        
    else:
        print("monte carlo integral incorrect for simple polynomials")
        failures += 1
    
    print("correct value is <{}>, functions value is <[{},{}]>"
            .format(correct,value[0]-value[1],value[0]+value[1]))
    print()
        
    print("function checked was 3*x^2")
    print("number of blocks and iterations was {}".format(n))
    print()
    
    def fun2(x): return np.sin(x)
    
    value = monte_carlo_integration(fun2, 0, 2, n, n)
    correct = 1.4161468365
    
    if correct <= value[0] + value[1] and correct >= value[0] - value[1]:
        print("monte carlo integral sum correct for a sine function")
        
    else:
        print("monte carlo integral sum incorrect for a sine function")
        failures += 1
    
    print("correct value is <{}>, functions value is <[{},{}]>"
            .format(correct,value[0]-value[1],value[0]+value[1]))
    print()
    print("function checked was sin(x)")
    print("number of blocks and iterations was {}".format(n))
    print()
    
    return failures

def main():
    
    print("------------------first derivative--------------------")
    failures = test_first_derivative()
    print()
    print("-----------------second derivative--------------------")
    failures += test_second_derivative()
    print()
    print("----------------riemann sum integral------------------")
    failures += test_num_riemann()
    print()
    print("----------------trapezoidal integral------------------")
    failures += test_num_trapezoid()
    print()
    print("------------------simpson integral--------------------")
    failures += test_num_simpson()
    print()
    print("----------------monte carlo integral------------------")
    failures += test_monte_carlo()
    
    print()
    print("------------------------------------------------------")
    print("Total of {} failures".format(failures))
    
    return 0


if __name__=="__main__":
    main()
