import numpy as np
import matplotlib.pyplot as plt
from num_calculus import *


def derivate_error_plot(x, dxmax=1, dxmin=1e-3, points=123, size = 15):
    """
    Plot derivate error 
    
    Param:
    x       -   point where calculate derivate
    dxmax   -   dx maximum value
    dxmin   -   dx minimum value
    points  -   how many points in plot
    size    -   figure size
    
    Return:
    None
    """
                
    fig, ax = plt.subplots(2, 1, figsize = (size,size))    
    dx = np.linspace(dxmax,dxmin, points)
        
    # First derivate tests    
    oikea2 = 6*x**2 + 2*x
    virhe2 = np.abs(first_derivate(testifun2, x, dx) - oikea2)
        
    oikea3 = 2* np.exp(2*x) + np.cos(x)        
    virhe3 = np.abs(first_derivate(testifun3, x, dx) - oikea3)
    # Plotting
    ax[0].plot(dx, virhe2, 'ro' ,label= r'$f(x) = 2x^3 + x^2 + 42 $ ')
    ax[0].plot(dx, virhe3, 'bx' ,label= r'$f(x) = exp(2x) + sin(x) $ ')
    ax[0].set_xlim(dx[0], dx[-1])
    ax[0].set_xscale('log')    
    ax[0].legend(loc=0)
    ax[0].set_xlabel("dx")
    ax[0].set_ylabel("Abs. error")
    ax[0].set_title("First derivate")
    
    # Second derivate test
    oikea2 = 12*x+2
    virhe2 = np.abs(second_derivate(testifun2, x, dx) - oikea2)
        
    oikea3 =4*np.exp(2*x) - np.sin(x)
    virhe3 = np.abs(second_derivate(testifun3, x, dx) - oikea3)
    
    ax[1].plot(dx, virhe2, 'ro' ,label= r'$f(x) = 2x^3 + x^2 + 42 $ ')
    ax[1].plot(dx, virhe3, 'bx' ,label= r'$f(x) = exp(2x) + sin(x) $ ')
    ax[1].set_xlim(dx[0], dx[-1])
    ax[1].set_xscale('log')
    ax[1].legend(loc=0)
    ax[1].set_xlabel("dx")
    ax[1].set_ylabel("Abs. error")
    ax[1].set_title("Second derivate")
        
    fig.savefig('derivaatta_virhe.pdf', dpi = 200)
    

def integral_error_plot(xmin = 0, xmax = 2, dxmin = 1e-3, dxmax = 1e-1, points = 123, size = 15):
    """
    Plot integral error for trapezoid 
    
    Param:
    xmin    -   min x value for integral starting point
    xmax    -   max x value for integral ending point
    dxmin   -   minimum value for dx
    dxmax   -   maximum value for dx
    points  -   how many grid points
    size    -   figure size
    
    Return:
    None
    """    
    
    fig = plt.figure(figsize = (size,size))    
    dx = np.linspace(dxmax, dxmin, points)
    
    oikeaF2  =  (1/2 * xmax**4 + 1/3 * xmax**3 + 42*xmax ) - ( 1/2 * xmin**4 + 1/3 * xmin**3 + 42*xmin )
    virhe2 = []
    for i in range(points):        
        x = np.linspace(xmin, xmax,  int((xmax-xmin)/dx[i]) )
        integral = trapezoid_int(x, testifun2)
        virhe2.append(np.abs(oikeaF2-integral))
    plt.plot(dx, virhe2, 'ro', label = r'$f(x) = 2x^3 + x^2 + 42 $ ')
    
    
    oikeaF3 = ( 1/2 * np.exp(2*xmax) - np.cos(xmax) ) - ( 1/2 * np.exp(2*xmin) - np.cos(xmin) )
    virhe3 = []
    for i in range(points):        
        x = np.linspace(xmin, xmax,  int((xmax-xmin)/dx[i]) )
        integral = trapezoid_int(x, testifun3)
        virhe3.append(np.abs(oikeaF3-integral))
    plt.plot(dx, virhe3, 'xb', label = r'$f(x) = exp(2x) + sin(x) $ ')
    
    plt.xlabel("dx")
    plt.ylabel("Abs. error")
    plt.title("Integral error with trapedoiz")
    plt.xlim(dx[0], dx[-1])
    plt.xscale('log')
    plt.legend(loc = 0)
    
    plt.savefig('integraalivirhe.pdf', dpi = 200)
    
def main():
    
    derivate_error_plot(1)
    integral_error_plot()
    
    
if __name__ == "__main__":
    main()