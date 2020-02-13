""" 
---------- EXERCISE 5 - problem 4 -----------
----- FYS-4096 - Computational Physics ------

Solves the potential grid created by two capacitors and
electric field associated with it

:function: sor           creates an estimate for the partial with sor method
                         with ignoring the capacitors
:function: gradient      solves the 2d gradient for every point in grid
:function: main          creates the plots and calls everything needed                         
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def sor(phi, rho, h, N, X, Y, omega=1.7):
    """
    Algorithm to solve the 2d partial differential 
    new estimation is calculateed with sor method
    see equations in for example in FYS-4096 materials
    :param: phi    last estimation
    :param: rho    potential function
    :param: h      grid separation
    :param: N      grid size
    :param: X      meshgrid of x
    :param: Y      meshgrid of Y
    :param: omega  constant for sor method
    
    :return: new estimation of the solution
    """
    phi2 = 1.0 * phi
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            # Do not update values that are at capacitor plates
            if not(((X[i, j] < -0.29) & (-0.31 < X[i, j]) & (-0.51 <= Y[i, j]) &\
               (Y[i, j] <= 0.51)) or (X[i, j] < 0.31) & (0.29 < X[i, j]) &\
               (-0.51 <= Y[i, j]) & (Y[i, j] <= 0.51)):
               
                phi2[i, j] = (1-omega)*phi2[i, j] + omega / 4.0 * \
                             (phi2[i + 1, j] + phi2[i - 1, j] + phi2[i, j + 1]+\
                             phi2[i, j - 1] + h ** 2 * rho[i, j])
            
    return phi2

def gradient(phi, h, N):
    """
    solve 2d gradient of the potential
    :param: phi    potential
    :param: h      grid separation
    :param: N      grid size
    
    :return: Ex, Ey gradients
    """
    
    # initalize array
    Ex = np.zeros((N, N))
    Ey = np.zeros((N, N))
    
    # gradient for every place on grid as seen on materials
    # of FYS-4096
    for i in range(1,N-1):
        for j in range(1, N-1):
            Ex[i, j] = (phi[i, j+1] - phi[i, j-1])/(2*h)
            Ey[i, j] = (phi[i+1, j] - phi[i-1, j])/(2*h)
            
        # edges of grid calculated separately
        Ex[i, 0] = (phi[i, 1] - phi[i, 0]) / h
        Ex[i, -1] = (phi[i, -1] - phi[i, -2]) / h
    for j in range(1, N-1):
        Ey[0, j] = (phi[1, j] - phi[0, j])/h
        Ey[-1, j] = (phi[-1, j] - phi[-2, j])/h
    return -Ex, -Ey

def main():
    N = 21
    
    # initalize grid
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    
    # initaliz epotential on grid
    phi = np.zeros((N, N))
    
    # create mesh of grid
    X, Y = np.meshgrid(x, y)
    
    # define spots of the capacitors on the potential grid (-1 and 1 potentials)
    # (needs big enough grid size i.e. >20)
    phi[ (X < -0.29) & (-0.31 < X) & (-0.51 <= Y) & (Y <= 0.51)] = -1
    phi[(X < 0.31) & (0.29 < X) & (-0.51 <= Y) & (Y <= 0.51)] = 1

    # tho function defined at every point to be zero
    rho = np.zeros((N, N))
    
    # gris size
    h = x[1]-x[0]

    tolerance = 1e-8
    
    # first estimation of potential
    phi2 = sor(phi, rho, h, N, X, Y)
    
    # run the potential estimation until tolerance condition is met
    loops = 1
    while np.max(np.abs(phi2-phi)) > tolerance:
        loops += 1
        phi = phi2
        phi2 = sor(phi, rho, h, N, X, Y)
    print(f"SOR-method looped {loops} times")

    # calculate the gradient
    Ex, Ey = gradient(phi, h, N)
    
    # plot the potential grid
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_wireframe(X, Y, phi, rstride=1, cstride=1)
    ax1.set_title("Potential in a box")
    
    # quiver plot the electric field
    ax2 = fig.add_subplot(122)
    ax2.quiver(X, Y, Ex, Ey)
    ax2.axis('equal')
    ax2.plot([-0.3, -0.3], [-0.5, 0.5], 'b-', linewidth=3)
    ax2.plot((0.3, 0.3), (-0.5, 0.5), 'r-', linewidth=3)
    ax2.set_title("Electric field in a box")

    plt.show()

if __name__ == "__main__":
    main()
