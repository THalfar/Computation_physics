from scipy.integrate import simps
import matplotlib.pyplot as plt
import numpy as np

"""CompPhys exercise 3 problem 1"""
# wolfram 1.57348

def int_fun(x,y):
    return (x+y)*np.exp(-1*np.sqrt(x**2+y**2))

def integral(grid_pts):
    """
    Computes and returns double integral as specified in the problem
    :param grid_pts:
    :return:
    """
    x0 = np.linspace(0,2,grid_pts)
    y0 = np.linspace(-2,2,grid_pts)
    val = np.empty([grid_pts,grid_pts])
    # Maybe use meshgrid
    for y in range(np.size(y0)):
        for x in range(np.size(x0)):
            val[x,y] = int_fun(x0[x],y0[y])
    sum = np.empty([grid_pts])
    # First sum
    for i in range(grid_pts):
        sum[i] = simps(val[i,],x0)
    tot_sum = 0
    # second sum
    for i in range(grid_pts):
        tot_sum = simps(sum,y0)
    return tot_sum

def main():
    """Plots integral sum value as a function of grid points"""
    grids = np.arange(4,81,2)
    sum = []
    for grid in grids:
        sum.append(integral(grid))
    plt.plot(grids,sum,'ro')
    plt.show()

main()