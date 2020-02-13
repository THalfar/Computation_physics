""" 
-------- EXERCISE 3 - problem 1 ------------
----- FYS-4096 - Computational Physics -----

integrates f(x) = (x+y)*exp(-sqrt(x^2+y^)) with 
numerical simspons method with many grid spacings.
Also plots the convergence of the integral.
"""

from scipy.integrate import simps
import numpy as np
import matplotlib.pyplot as plt


    
def main():
    
    # define the function to integrate
    def fun(x,y): return (x+y)*np.exp(-np.sqrt(x**2+y**2))
    
    # logarithmic array of amount intervals weighted to smaller numbers
    # and rounded to integers
    N = 10**(np.linspace(1,2,30)**2)
    N = np.round(N)
    I = np.zeros(30)
    i = 0
    
    print("Value of integral at given amount of intervals:")
    
    for n in N:
        # integral for each n
        # sizes of grids same for both directions
        
        X = np.linspace(-2,2,int(n))
        Y = np.linspace(0,2,int(n))
        J = np.zeros(int(n))
        
        j = 0
        
        for x in X:
            
            # integrate for each index x
            J[j] = simps(fun(x,Y),Y)
            j += 1
            
        # integrate the 'integral function' J(x) over X
        I[i] = simps(J,X)
        
        # display the value of the integral I(n)
        print("I = {}, n = {}".format(I[i],n))
        i += 1
        
    # plot I(n)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    ax1.semilogx(N,I)
    ax1.grid(True)
    
    #  axis labels
    ax1.set_xlabel('N')
    ax1.set_ylabel('integral')
    
    # change the last value to average of the last two so on the logarithmic
    # scale the last one doesn't go to -inf, just for representation
    I_last = I[-1]
    I[-1] = (I[-1]+I[-2])/2
    
    ax2.loglog(N,abs(I-I_last))
    
    #  axis labels
    ax2.set_xlabel('N')
    ax2.set_ylabel('convergence')
    
    title = 'convergence of simpson integral'
    fig.suptitle(title)
    
    ax2.grid(True)
    
    #show figure
    plt.show()
        
    
if __name__=="__main__":
    main()

