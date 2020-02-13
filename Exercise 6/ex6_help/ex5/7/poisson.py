"""
Solving the Poisson equation in 2D using the Jacobi, Gauss-Seidel, and SOR
update schemes. 
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def solve_jacobi(points):
    
    # x and y indices for iteration, not including boundary points.
    x = np.linspace(1, points.shape[0] - 2, points.shape[0] - 2)
    y = np.linspace(1, points.shape[1] - 2, points.shape[1] - 2)
    
    max_diff = 1
    iterations = 0
    
    # Repeating until maximum absolute difference is under 10^-10
    while max_diff > 1e-10:
        # Make new array for values from this iteration.
        new_points = np.zeros_like(points)
        new_points[:,:] = points[:,:]
        # For every non-boundary x and y
        for i in x:
            for j in y:
                i = int(i)
                j = int(j)
                new_points[i, j] = 1/4*(points[i+1,j] + points[i-1, j] + points[i, j+1] + points[i, j-1])

        max_diff = np.max(np.abs(points - new_points))
        iterations += 1
        # Replace the old values with the new ones.
        points = new_points
        
    return points, iterations

def solve_gauss_seidel(points):
    
    # x and y indices for iteration, not including boundary points.
    x = np.linspace(1, points.shape[0] - 2, points.shape[0] - 2)
    y = np.linspace(1, points.shape[1] - 2, points.shape[1] - 2)
    
    max_diff = 1
    iterations = 0
    
    # Repeating until maximum absolute difference is under 10^-10
    while max_diff > 1e-10:
        # Make new array for values from this iteration.
        new_points = np.zeros_like(points)
        new_points[:,:] = points[:,:]
        # For every non-boundary x and y
        for i in x:
            for j in y:
                i = int(i)
                j = int(j)
                new_points[i, j] = 1/4*(points[i+1,j] + new_points[i-1, j] + points[i, j+1] + new_points[i, j-1])
        
        max_diff = np.max(np.abs(points - new_points))
        iterations += 1
        # Replace the old values with the new ones.
        points = new_points
    
    return points, iterations

def solve_SOR(points):
    
    # x and y indices for iteration, not including boundary points.
    x = np.linspace(1, points.shape[0] - 2, points.shape[0] - 2)
    y = np.linspace(1, points.shape[1] - 2, points.shape[1] - 2)
    
    max_diff = 1
    iterations = 0
    
    # Repeating until maximum absolute difference is under 10^-10
    while max_diff > 1e-10:
        # Make new array for values from this iteration.
        new_points = np.zeros_like(points)
        new_points[:,:] = points[:,:]
        # For every non-boundary x and y
        for i in x:
            for j in y:
                i = int(i)
                j = int(j)
                new_points[i, j] = (1-1.8)*points[i,j] + 1.8/4*(points[i+1,j] + new_points[i-1, j] + points[i, j+1] + new_points[i, j-1])
        
        max_diff = np.max(np.abs(points - new_points))
        iterations += 1
        # Replace the old values with the new ones.
        points = new_points
    
    return points, iterations
    

def main():
    # Plotting graphs and printing the numbers of iterations.
    
    
    # Defining the boundary conditions.
    n = 20
    points = np.zeros((n,n))
    points[:,0] = 1
    
    # Points for the plot.
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,n)
    X,Y = np.meshgrid(x, y)
    
    # Plotting the initial conditions.
    print("Initial conditions:")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X,Y,points,rstride=1,cstride=1)
    plt.show()
    
    print("Jacobi method:")
    jacobi_solution, iter_j = solve_jacobi(points)
    print("Iterations:", iter_j)
    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.plot_wireframe(X,Y,jacobi_solution,rstride=1,cstride=1)
    plt.show()
    
    print("Gauss-Seidel method:")
    GS_solution, iter_GS = solve_gauss_seidel(points)
    print("Iterations:", iter_GS)
    fig = plt.figure()
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.plot_wireframe(X,Y,GS_solution,rstride=1,cstride=1)
    plt.show()
    
    print("SOR method:")
    SOR_solution, iter_SOR = solve_SOR(points)
    print("Iterations:", iter_SOR)
    fig = plt.figure()
    ax4 = fig.add_subplot(111, projection='3d')
    ax4.plot_wireframe(X,Y,SOR_solution,rstride=1,cstride=1)
    plt.show()


if __name__=="__main__":
    main()