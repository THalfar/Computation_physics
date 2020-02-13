import numpy as np
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D

def gauss_seidel(f, boundary_x, boundary_y):
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
                # if not boundary condition calculate else just move to next
                if i not in boundary_x or j not in boundary_y:
                    next_iter[i, j] = 1 / 4 * (f[i + 1, j] + next_iter[i - 1, j] + f[i, j + 1] + next_iter[i, j - 1])
        # check if largest change form last iter is smaller than tolerance
        if tolerance > np.amax(np.abs(f-next_iter)):
            inaccurate = False
        f = copy.deepcopy(next_iter)
    return next_iter, counter

def initialize(xx,yy):
    """
    initial conditions for our problem and boundary conditions in the middle of grid
    :param xx: x-coordinates
    :param yy: y-coordinates
    :return:
    """
    f = np.zeros((len(xx),len(yy)))
    boundary_x = [7, 13]
    boundary_y = []
    for i in range(5,15):
        boundary_y.append(i)
        f[7,i] = -1
        f[13, i] = 1
    return f, boundary_x, boundary_y

def main():
    # initializing grid
    xx = np.linspace(-1, 1, 21)
    yy = np.linspace(-1, 1, 21)
    X, Y = np.meshgrid(xx, yy)
    f, boundary_x, boundary_y  = initialize(xx, yy)

    # calculate and plot potential
    f, counter = gauss_seidel(f, boundary_x, boundary_y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, f, rstride=1, cstride=1)

    # calculating electric field with np.gradient and plotting it with quiver
    fig2 = plt.figure()
    Ex, Ey = np.gradient(f,xx ,yy)
    Ex = -1 * Ex
    Ey = -1 * Ey
    fig1d = plt.quiver(X, Y, Ex, Ey)

    plt.show()

if __name__ == "__main__":
    main()