""" 
-------- EXERCISE 3 - problem 4 ------------
----- FYS-4096 - Computational Physics -----

integrates electric field produced by 1-D rod 
dE = 1/(4pi epsilon0) * Qdx/Lr^2 at point (1.1,0) 
and compares it to value given by analytically solved value
also plots the electric field around the rod with quiver

integral used is the numerical simpson method created
in the first exercise set copied here.
"""

import numpy as np
import matplotlib.pyplot as plt

def efield_num(x,y,L = 2,Q = (4*np.pi*8.85e-12)**(-1)):
    """
    :param: x         : x-coord to be calculated at
    :param: y         : y-coord to be calculated at
    :param: L = 2     : Length of rod
    :param: Q = a     : Charge of rod a = 1/(4*pi*8.85e-12)
    
    :return: Ex, Ey   : electric field components at x,y
    """
    a = (4*np.pi*8.85e-12)**(-1)
    
    # remove division by 0
    # define dE at both axis directions and integrate both over the rod
    if y != 0:
        def dEx(xx): return (Q/(a*L))*(np.sqrt(y**2+(x-xx)**2))**(-2)*np.sin(np.arctan2((x-xx),y))
        def dEy(xx): return (Q/(a*L))*(np.sqrt(y**2+(x-xx)**2))**(-2)*np.cos(np.arctan2((x-xx),y))
        Ey = num_simpson(dEy,-L/2,L/2,10000)
    else:
        def dEx(xx): return (Q/(a*L))*(np.sqrt(y**2+(x-xx)**2))**(-2)
        def dEy(xx): return 0
        Ey = 0
    
    Ex = num_simpson(dEx,-L/2,L/2,10000)
    return Ex,Ey
    
def efield_real(x,y,L = 2,Q = (4*np.pi*8.85e-12)**(-1)):
    """
    :param: x         : x-coord to be calculated at
    :param: y         : y-coord to be calculated at
    :param: L = 2     : Length of rod
    :param: Q = a     : Charge of rod a = 1/(4*pi*8.85e-12)
    
    :return: Ex, Ey   : electric field components at x,y
    """
    a = (4*np.pi*8.85e-12)**(-1)
    
    # analytically solved answer at x - axis (y - axis 0)
    return (Q/(a*L))*((1/(x-L/2))-(1/(x+L/2))), 0

def num_simpson(function, a, b, n ):
    """
    :param: function : function to integrate
    :param: a        : lower bound of integral
    :param: b        : higher bound of integral
    :param: n        : number of intervals > 1, must be even!
    :return: simpson numerical integral

     calculates simpson instegral sum as:
     I = sum(1/3*(f(2x)+4*f(2x+dx)+f(2x+2dx)) * dx) 
       = 1/3 * sum(f(2x)+4*f(2x+dx)+f(2x+2dx)) * dx
     the grid is set as uniform
    """
    
    x = np.linspace(a,b,n+1)
    
    return 1/3 * sum(function(x[0:n-1:2])+4*function(x[1:n:2])+function(x[2:n+1:2])) * (b-a)/n

def main():

    print('Electric field of dE = 1/(4pi epsilon0) * Qdx/Lr^2')
    print('integrated over the length of the rod at (1.1,0)')
    print()
    print('Numerically solved electric field:  {}'.format(efield_num(1.1,0)))
    print('Analytically solved electric field: {}'.format(efield_real(1.1,0)))
    
    # solve the field around the rod in 40 by 40 grid 
    # grid
    X, Y = np.meshgrid(np.linspace(-1.2, 1.2, 40), np.linspace(-1.2, 1.2, 40))
    
    # x and y (U,V) components of electric field at grid points
    U = np.zeros(X.shape)
    V = np.zeros(Y.shape)
    
    for i in range(0,39):
        for j in range(0,39):
            U[i,j], V[i,j] = efield_num(X[i,j],Y[i,j])
    
    # plot with quiver
    fig, ax = plt.subplots()
    ax.set_title("Electric field")
    M = np.hypot(U, V)
    Q = ax.quiver(X, Y, U, V, M, units='x', width=0.011)
    qk = ax.quiverkey(Q, 0.9, 0.9, 1,r'$1$', labelpos='E', coordinates='figure')
    ax.scatter(X, Y, color='0.5', s=1)

    plt.show()

if __name__=="__main__":
    main()

