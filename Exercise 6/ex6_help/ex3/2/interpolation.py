""" 
-------- EXERCISE 3 - problem 2 ------------
----- FYS-4096 - Computational Physics -----

interpolates f(x) = (x+y)*exp(-sqrt(x^2+y^)) with spline_class.py
along line y = sqrt(1.75)x and compares the interpolation with
original values with the lower grid res and higher grid res
Those are then plotted

"""
import numpy as np
import matplotlib.pyplot as plt
from spline_class import *

def main():
    
    # define the function to interpolate
    def fun(x,y): return (x+y)*np.exp(-np.sqrt(x**2+y**2))
    
    # original grid
    X = np.linspace(-2,2,30)
    Y = np.linspace(-2,2,30)
    
    # function valued at every grid point
    f = np.zeros((X.size,Y.size))
    i = 0
    j = 0
    for x in X:
        for y in X:
            f[i,j] = fun(x,y)
            j+=1
        i+=1
        j = 0
    
    # spline interpolation
    spl = spline(x=X, y=Y, f=f, dims = 2)
    
    # line along which interpolation and comparing happens
    y = np.linspace(0,2,100)
    x = np.sqrt(1.75)*np.linspace(0,2,100)
    
    # create estimate vector
    estimate = np.zeros(x.size)
    i = 0
    for xx in x:
        estimate[i] = spl.eval2d(xx,y[i])
        i+=1
    
    # real function at the line
    real = fun(x,y)
    
    # the line of the original spacing
    xx = np.linspace(0,2,15)
    yy = np.sqrt(1.75)*np.linspace(0,2,15)
    
    # non interpolated
    non_intrpl = fun(xx,yy)
    
    # plot the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(np.sqrt(xx**2+yy**2), non_intrpl, 'b', label='non_interpolated')
    ax.plot(np.sqrt(x**2+y**2),real,'g', label='real')
    ax.plot(np.sqrt(x**2+y**2),estimate,'r', label='estimate')
    
    plt.legend(loc='upper right',
           ncol=1, borderaxespad=0.)
    
    #  axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('f')
    
    title = 'non interpolation vs real vs interpolated'
    fig.suptitle(title)
    
    # show the figure
    plt.show()
    

if __name__=='__main__':
    main()
