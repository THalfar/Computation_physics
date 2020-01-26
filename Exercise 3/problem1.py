import numpy as np
from scipy.integrate import simps, dblquad
import matplotlib.pyplot as plt

# Function that is integrated
def fun(x,y): return (x+y)*np.exp(-np.sqrt(x**2 + y**2))


def integraalivirhe():

    right = dblquad(fun, 0, 2, -2, 2) # right integral value using scipy dlbquad
    grids = np.geomspace(2,1000, 30, dtype = int) #grid 
    error = [] # error of integral

    for i in grids:
        x = np.linspace(0,2, i)
        y = np.linspace(-2,2,i)                    
        X,Y = np.meshgrid(x,y)
        
        Z = fun(X,Y)
        x_int = simps(Z, x) # x-axis simpsons integral
        simps_int = simps(x_int, y) # y-axis simpsons integral
    
        error.append(np.abs(simps_int-right[0])) # append to error
        
    
    plt.plot(grids, error, 'or')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Error of simpsons integral with different grids")
    plt.ylabel("Abs error")
    plt.xlabel("Grid size")
    
def main():
    integraalivirhe()
    
if __name__=="__main__":
    main()
    