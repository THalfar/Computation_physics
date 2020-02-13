import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy

def jacobi(f):
    """
    solves poissons equation with jacobi method
    :param f: function values at start
    :return: function values at end and number of iterations
    """
    # copy to keep data from last iter
    next_iter = copy.deepcopy(f)
    tolerance = 0.0001
    inaccurate = True
    counter = 0
    while inaccurate:
        counter += 1
        # jacobi method from week 5 lecture formula 61
        for i in range(1,len(f[0,:])-1):
            for j in range(1,len(f[0, :])-1):
                next_iter[i,j] = 1/4*(f[i+1,j]+f[i-1,j]+f[i,j+1]+f[i,j-1])
        # check if largest change form last iter is smaller than tolerance
        if tolerance > np.amax(np.abs(f-next_iter)):
            inaccurate = False
        f = copy.deepcopy(next_iter)
    return next_iter, counter

def gauss_seidel(f):
    """
    solves poissons equation with gauss-seidel method
    :param f: function values at start
    :return: function values at end and number of iterations
    """
    # copy to keep data from last iter
    next_iter = copy.deepcopy(f)
    tolerance = 0.0001
    inaccurate = True
    counter = 0
    while inaccurate:
        counter += 1
        # Gauss-Seidel method from week 5 lecture formula 62
        for i in range(1,len(f[0,:])-1):
            for j in range(1,len(f[0, :])-1):
                next_iter[i,j] = 1/4*(f[i+1,j]+next_iter[i-1,j]+f[i,j+1]+next_iter[i,j-1])
        # check if largest change form last iter is smaller than tolerance
        if tolerance > np.amax(np.abs(f-next_iter)):
            inaccurate = False
        f = copy.deepcopy(next_iter)
    return next_iter, counter

def sor(f, w):
    """
    solves poissons equation with SOR method
    :param f: function values at start
    :return: function values at end and number of iterations
    """
    # copy to keep data from last iter
    next_iter = copy.deepcopy(f)
    tolerance = 0.001
    inaccurate = True
    counter = 0
    while inaccurate:
        counter += 1
        # SOR method from week 5 lecture formula 63
        for i in range(1,len(f[0,:])-1):
            for j in range(1,len(f[0, :])-1):
                next_iter[i,j] =(1-w)*f[i,j]+w/4*(f[i+1,j]+next_iter[i-1,j]+f[i,j+1]+next_iter[i,j-1])
        # check if largest change form last iter is smaller than tolerance
        if tolerance > np.amax(np.abs(f-next_iter)):
            inaccurate = False
        f = copy.deepcopy(next_iter)
    return next_iter, counter

def initialize(xx,yy):
    """
    initial conditions for our problem
    :param xx: x-coordinates
    :param yy: y-coordinates
    :return:
    """
    f = np.zeros((len(xx),len(yy)))
    for i in range(len(xx)):
        f[i,0] = 1
    return f

def main():
    # initializing grid
    xx = np.linspace(0, 1, 20)
    yy = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(xx, yy)
    f = initialize(xx, yy)

    # using jacobi method and showing results
    f1, counter = jacobi(f)
    print("Jacobi:")
    print("number of iterations: ", counter)
    print("value in the middle of grid: ", f1[10, 10])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, f1, rstride=1, cstride=1)

    # using Gauss-Seidel method and showing results
    f2, counter = gauss_seidel(f)
    print("Gauss-Seidel")
    print("number of iterations: ", counter)
    print("value in the middle of grid: ", f2[10, 10])
    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, f2, rstride=1, cstride=1)

    # using SOR method and showing results
    f3, counter = sor(f, 1.8)
    print("SOR")
    print("number of iterations: ", counter)
    print("value in the middle of grid: ", f3[10,10])
    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, f3, rstride=1, cstride=1)

    plt.show()

if __name__ == "__main__":
    main()
