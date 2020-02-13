import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

def interpolation_error():
    # Regularly-spaced, coarse grid
    n_data = 30
    xmax, ymax = 2, 2 # limits of grid
    x = np.linspace(-xmax, xmax, n_data)
    y = np.linspace(-ymax, ymax, n_data)
    X, Y = np.meshgrid(x, y)        
    Z = (X+Y) * np.exp(-np.sqrt(X**2+Y**2))
    f = interp2d(x, y, Z, kind='cubic') # using scipy method of interpolation, faster as c++ code?
    # Calculate the xy values where interpolation is calculated    
    n_int = 100
    x_inter = np.linspace(0,2/ np.sqrt(1.75),n_int)
    y_inter = np.sqrt(1.75) * x_inter
    
    right_val = [] # right values at points
    interpol_val = [] # interpolation values at point
    # loop over line values
    for i in range(n_int):
        right = (x_inter[i]+y_inter[i]) * np.exp(-np.sqrt(x_inter[i]**2 + y_inter[i]**2))
        right_val.append(right) 
        
        inter = f(x_inter[i],y_inter[i])
        interpol_val.append(inter)
        
    # plot the interpolated values vs true value of function    
    plt.plot(x_inter, right_val, 'ro')
    plt.plot(x_inter, interpol_val, 'b-')
    plt.title("Interpolation value vs true value plot")
    plt.xlabel("x-point value")
    plt.ylabel("Value of function")
    plt.show()

def main():
    interpolation_error()
    
if __name__=="__main__":
    main()
    