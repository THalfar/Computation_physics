""" 
-------- EXERCISE 2 - problem 3 ------------
----- FYS-4096 - Computational Physics -----

steepest_descent.py contains a simple steepest descent method to find the
minimum of a function nearby given starting point

"""
import num_calculus as nc
import numpy as np
import matplotlib.pyplot as plt

def minimum(fun, x, xl = 0, dl = 0):
    """
    :param: fun    is the function to be checked
    :param: x      is the value around which the ectremum is looked for
    :param: xl     is last x value used in recursion
    :param: dl     is the last derivative used in recursion
    
    :return: itself or extremum and corresponding x (prints the outcome)
    
    calculates recursively the minimum of the function
    near the given point x
    """
    
    derivative = nc.first_derivative(fun, x, 0.001)
    
    gamma = -abs((x-xl)/(derivative-dl))
    
    
    
    if abs(derivative) > 1e-6:
    
        return minimum(fun, x + gamma*derivative,x,derivative)
        
    else:
        print("The minimum for the function is {} at x = {}"
              .format(fun(x), x))
        return x,fun(x)
              
def test_minimum():
    """ tests minimum function"""
    
    
    def fun(x): return x**4+x**2+x-3
    orig = 1
    min_x,min_f = minimum(fun, orig)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xx = np.linspace(-2,2,100)
    ax.plot(xx,fun(xx))
    ax.plot(min_x,min_f, marker='o', label='minimum')
    ax.plot(orig,fun(orig), marker='o', label='original point')
    
    plt.legend(loc='upper right',
           ncol=1, borderaxespad=0.)
    
    #  axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('f')
    
    title = 'minimum of x*sin(x)'
    fig.suptitle(title)
    
    name = 'minimum.pdf'
    
    # save and show figures
    fig.savefig(name,dpi=200)
    plt.show()
    
    def fun(x): return x*np.sin(x)
    orig = 3
    min_x,min_f = minimum(fun, orig)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xx = np.linspace(2,7,100)
    ax.plot(xx,fun(xx))
    ax.plot(min_x,min_f, marker='o', label='minimum')
    ax.plot(orig,fun(orig), marker='o', label='original point')
    
    plt.legend(loc='upper right',
           ncol=1, borderaxespad=0.)
    
    #  axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('f')
    
    title = 'minimum of x*sin(x)'
    fig.suptitle(title)
    
    name = 'minimum.pdf'
    
    # save and show figures
    fig.savefig(name,dpi=200)
    plt.show()
    
def main():
    
    test_minimum()
    
    
if __name__=="__main__":
    main()
