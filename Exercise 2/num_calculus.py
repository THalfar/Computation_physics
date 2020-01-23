import numpy as np
import matplotlib.pyplot as plt

def first_derivate(function, x, dx):
    """
    Calculates first derivate 
    
    Param:
        function  -  Function that shall be derivated
        x         -  Where derivate
        dx        -  Difference step in derivate calculation  
    
    Return:
        Value of first derivate
    """
    
    jakaja = 2*dx
    jaettava = function(x+dx) - function(x-dx)
    
    return jaettava / jakaja

def second_derivate(function, x, dx):
    """
    Calculates second derivate 
    
    Param:
        function  -  Function that shall be derivated
        x         -  Where derivate
        dx        -  Difference step in derivate calculation  
    
    Return: 
        Value of second derivate    
    """
    
    jakaja = dx**2
    jaettava = function(x+dx) + function(x-dx) - 2*function(x)
    
    return jaettava / jakaja


def trapezoid_int(x, function):
    """
    Calculates a numerical integral of the trapezoid integral
    
    Param:
        x       -  a uniformly spaced x-values 
        function -  function thats going to be integrated
    
    Return:
        Trapezoid integral of function
    """
    
    # calculate point interval, assuming that x has at least 2 points    
    h = x[1] - x[0]
        
    summa = 0
    
    # Loops through points x and add to the summa the block values
    for i in range(0, len(x)-1):
        
        summa += (function(x[i]) + function(x[i+1]))*h
        
    return summa * 0.5

 
def riemann_sum(x, function):
    """   
    Calculates numerical integral using rieman sum
    
    Param:    
        x       -  a uniformly spaced x-values 
        function -  function thats going to be integrated
    
    Return:
        Rieman sum integral of function 
    """   
    
    # calculate point interval, assuming that x has at least 2 points    
    h = x[1] - x[0]
        
    summa = 0
    
    # Loops through points x and add to the summa the block values
    for i in range(0, len(x)-1):
        
        summa += (function(x[i+1]))*h
        
    return summa


def simpson_int(x, function):
    """
    Calculates numerical integral using simpson rule
    Param:    
        x       -  a uniformly spaced x-values 
        function -  function thats going to be integrated
    Return:
        simpson integral of function
    """
  
    
    # calculate point interval, assuming that x has at least 2 points    
    h = x[1] - x[0]

    summa = 0    
    # Number of intervals is one shorter than len(x)                   
    askelia = int( (len(x) -1) / 2  )
        
    for i in range(askelia): # range(askelia) = n-1
        summa += function(x[2*i]) + 4*function(x[2*i+1]) + function(x[2*i+2])    
    summa *= h/3
        
    # Add the odd tail 
    if (len(x)-1) % 2 != 0:  
        summa += h/12 * ( -1*function(x[-3]) + 8*function(x[-2]) + 5*function(x[-1]) )
                    
    return summa 


def monte_carlo_integration(fun,xmin,xmax,blocks,iters):
    """
    Monte carlo integral
    
    Param:
        fun    -   Function that shall be integrated
        xmin   -   Minimum value of x
        xmax   -   Maximum value of x
        blocks -   How many blocks in monte carlo integration
        iters  -   Number of iterations per block
    
    Return:
        Monte carlo intergral of function
    """     
    # Set blocks and Lenght of each
    block_values=np.zeros((blocks,)) 
    L=xmax-xmin
    
    # Iterate through blocks and calculate function values at random points in block
    for block in range(blocks):
        for i in range(iters):
            
            x = xmin+np.random.rand()*L
            block_values[block]+=fun(x) # sum in block the values at interval random points
            
        block_values[block]/=iters 
            
    I = L*np.mean(block_values) # use mean of block value to calculate integral
    dI = L*np.std(block_values)/np.sqrt(blocks) # take std. to measure statistical error
            
    return I,dI

def gradient(fun, point, h):
    """
    Calculates gradient at point assuming function have same input as
    dimension of point

    Parameters
    ----------
    fun : function
        function which gradient need calculating
    point : np.array
        point where calculate gradient
    h : double
        differential length for differencial calculationg

    Returns
    -------
    np.array of gradient values

    """
    
    result = np.zeros(len(point))
    
    for i in range(len(point)):
        askelTaakse = np.copy(point)
        askelTaakse[i] -= h        
        askelEteen = np.copy(point)
        askelEteen[i] += h
        jakaja = 2*h        
        jaettava = fun(askelEteen) - fun(askelTaakse)
        result[i] = jaettava / jakaja
            
    return result
        
def gradtest2D():
    t1 = np.array([1.,1.])    
    h_values = np.geomspace(1e-1,1e-5,42)
    errors1 = []
    true1 = np.array([2*t1[0]*np.exp(t1[0]**2+t1[0]**2), 2*t1[1]*np.exp(t1[0]**2+t1[1]**2)])    
    
    for h in h_values:        
        testi1 = gradient(gradfun1, t1, h)        
        errors1.append(np.linalg.norm(true1-testi1))
    
        
    fig = plt.figure()
    plt.plot(h_values,errors1, 'or', label= r'$f(x,y) = exp(x^2+y^2) $')        
    plt.xlabel("h")
    plt.ylabel("L2 error")
    plt.title("Grad error 2D")    
    plt.xlim(h_values[0], h_values[-1])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc = 0)
    fig.savefig('grad_2D.pdf', dpi = 200)
    
     
def gradtest3D():
    p = np.array([1., 1., 1.])   
    h_values = np.geomspace(1e-1,1e-5,42)
    errors = []
    true = np.array([p[0]*np.exp(p[1]**2+p[2]**2), p[0]*(2*p[1]**2+1)*np.exp(p[1]**2+p[2]**2), 
                     2*p[0]*p[1]*p[2]*np.exp(p[1]**2+p[2]**2)])
    
    for h in h_values:
        test = gradient(gradfun2, p, h)
        errors.append(np.linalg.norm(true-test))
    
    fig = plt.figure()
    plt.plot(h_values,errors, 'or', label= r'$f(x,y,z) = x*y*exp(y^2+z^2) $')        
    plt.xlabel("h")
    plt.ylabel("L2 error")
    plt.title("Grad error 3D")    
    plt.xlim(h_values[0], h_values[-1])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc = 0)
    fig.savefig('grad_3D.pdf', dpi = 200)        
    
    
def SGD(fun, point, iterations, a, h):
    """
    Steepest Gradient Descent algorithm

    Parameters
    ----------
    fun : function
        function what to minimize
    point : np.array
        starting point
    iterations : int
        Number of iterations
    a : double
        scaling value for gamma 
    h : double
        h value for gradient

    Returns
    -------
    point : np.array
        minimum where SGD descents

    """
        
    for n in range(iterations):        
        gamma = a / (np.linalg.norm(gradient(fun, point, h))+1)        
        point = point - gamma * gradient(fun, point, h)
    return point
        
# Simple test for SGD
def SGDtest():
    
    point = np.array([-.42, 4.2])
    h = 1e-6
    a = 1e-2
    # Testing 2D case minimum is 0    
    tulos = SGD(mintest, point, int(1e4), a, h)    
    print("SGD gives on mintest:{} with value: {} a: {}".format(tulos,mintest(tulos), a))
            
    # Testing 3D case minimum is 1
    point = np.array([-4.2,3.2,1.2])
    tulos = SGD(mintest2, point, int(1e4), a, h)        
    print("SGD gives on mintest2:{} with value: {} a: {}".format(tulos,mintest2(tulos), a))
    
    
    a_values= np.geomspace(5e-3,1e-6,13) # tosi herkkä a arvoille..    
    for a in a_values:
        # Testing 3D case minimum is -1
        point = np.array([-2.,3.,-4.])
        tulos = SGD(mintest2, point, int(1e4), a, h)        
        print("SGD gives on mintest3:{} with value: {} a: {}".format(tulos,mintest3(tulos), a))
    
    
    
"""
These are example functions for testing
"""
# f(x) = x^2 + 3x
def testifun1(x): return x**2 + 3*x

# f(x) = 2x^3+xˆ2+42
def testifun2(x): return 2* x**3 + x**2 + 42

# f(x) = exp(2x) + sin(x)
def testifun3(x): return np.exp(2*x) + np.sin(x)

def gradfun1(x): return np.exp((x[0])**2 + (x[1])**2)

def gradfun2(x): return x[0]*x[1]*np.exp(x[1]**2 + x[2]**2)

def mintest(x): return (x[0]-3)**2 + (x[1]+2)**2

def mintest2(x): return np.exp(x[0]**2 + x[1]**2 + x[2]**2 )

def mintest3(x): return np.sin(x[0] + x[1] + x[2] )


def test_first_derivate(x, dx):
   """
   Testing first derivate with three function
   
   Param:
       x   -  point where derivate
       dx  -  difference step in derivate
   Return:
       none
   
   """
   # f1(x) = x^2 + 3x -> 2x + 3
   oikea = 2*x+3
   testi = first_derivate(testifun1, x, dx)
   erotus = np.abs(oikea-testi)
   print("f1'({}) dx: {} virhe abs.: {}".format(x,dx,erotus))
   
   # f2(x) = 2x^3+xˆ2+42 -> 6xˆ2 + 2x   
   oikea =  6*x**2 + 2*x
   testi = first_derivate(testifun2, x, dx)
   erotus = np.abs(oikea-testi)
   print("f2'({}) dx: {} virhe abs.: {}".format(x, dx, erotus))
       
   # f3(x) = exp(2x) + sin(x) -> 2*exp(2x) - cos(x)
   oikea = 2* np.exp(2*x) + np.cos(x)
   testi = first_derivate(testifun3, x, dx)
   erotus = np.abs(oikea-testi)
   print("f3'({}) dx: {} virhe abs.: {}".format(x, dx, erotus))


def test_second_derivate(x, dx):
    """
    Testing second derivate with three function
    
    Param:
        x   -  point where derivate
        dx  -  difference step in derivate
    
    Return:
        None
    """
        
    # f1(x) = x^2 + 3x -> 2
    oikea = 2
    testi = second_derivate(testifun1, x, dx)
    erotus = np.abs(oikea-testi)
    print("f1''({}) dx: {} virha abs: {}".format(x,dx,erotus))
    
    # f2(x) = 2x^3+xˆ2+42 -> 12x + 2  
    oikea = 12*x+2
    testi = second_derivate(testifun2, x, dx)
    erotus = np.abs(oikea-testi)
    print("f2''({}) dx: {} virhe abs: {}".format(x,dx,erotus))
    
    # f3(x) = exp(2x) + sin(x) -> 2*exp(2x) - sin(x)
    oikea = 4*np.exp(2*x) - np.sin(x)
    testi = second_derivate(testifun3, x, dx)
    erotus = np.abs(oikea-testi)
    print("f3''({}) dx: {} virhe abs: {}".format(x,dx,erotus))
   

def test_integral(jako, xmin, xmax, blocks = 100, iterations = 100):
    """
    Testing of numerical integrals
    
    Param:
        jako   -   how many points if integral method use these
        xmin   -   minimum x of test integrals
        xmax   -   maximum x of test integrals
        blocks -   number of blocks in monte carlo integral
        iterations - number of iterations per block in monte carlo   
    Return:
        None 
    """
                    
    x = np.linspace(xmin,xmax, jako)
    jako = jako -1 # montako väliä
    
    # Right anwers analytically 
    
    # f1(x) = x^2 + 3x -> 1/3 * x^3 + 3/2 * x^2
    oikeaF1 = ( 1/3 * xmax**3 + 3/2 * xmax**2 ) -  ( 1/3 * xmin**3 + 3/2 * xmin**2 ) 
    
    # f2(x) = 2x^3+xˆ2+42 -> 1/2 * x^4 + 1/3 x^3 + 42x 
    oikeaF2 = (1/2 * xmax**4 + 1/3 * xmax**3 + 42*xmax ) - ( 1/2 * xmin**4 + 1/3 * xmin**3 + 42*xmin )
    
    # f3(x) = exp(2x) + sin(x) -> 1/2 exp(2x) - cos(x)
    oikeaF3 = ( 1/2 * np.exp(2*xmax) - np.cos(xmax) ) - ( 1/2 * np.exp(2*xmin) - np.cos(xmin) )
        
    testi = riemann_sum(x, testifun1)
    erotus = np.abs(oikeaF1-testi)
    print("Rieman: F1({} , {}) jako: {} virhe abs.: {}".format(xmin, xmax, jako , erotus))
                
    testi = riemann_sum(x, testifun2)
    erotus = np.abs(oikeaF2-testi)
    print("Rieman: F2({} , {}) jako: {} virhe abs.: {}".format(xmin, xmax, jako , erotus))
    
    testi = riemann_sum(x, testifun3)
    erotus = np.abs(oikeaF3-testi)
    print("Rieman: F3({} , {}) jako: {} virhe abs.: {}".format(xmin, xmax, jako , erotus))
    
    testi = trapezoid_int(x, testifun1)
    erotus = np.abs(oikeaF1-testi)
    print("Trapezoid: F1({} , {}) jako: {} virhe abs.: {}".format(xmin, xmax, jako , erotus))
            
    testi = trapezoid_int(x, testifun2)
    erotus = np.abs(oikeaF2-testi)
    print("Trapezoid: F2({} , {}) jako: {} virhe abs.: {}".format(xmin, xmax, jako , erotus))
            
    testi = trapezoid_int(x, testifun3)
    erotus = np.abs(oikeaF3-testi)
    print("Trapezoid: F3({} , {}) jako: {} virhe abs.: {}".format(xmin, xmax, jako , erotus))
            
    testi = simpson_int(x, testifun1)
    erotus = np.abs(oikeaF1-testi)
    print("Simpson: F1({} , {}) jako: {} virhe abs.: {}".format(xmin, xmax,jako, erotus))
        
    testi = simpson_int(x, testifun2)
    erotus = np.abs(oikeaF2-testi)
    print("Simpson: F2({} , {}) jako: {} virhe abs.: {}".format(xmin, xmax,jako, erotus))
               
    testi = simpson_int(x, testifun3)
    erotus = np.abs(oikeaF3-testi)
    print("Simpson: F3({} , {}) jako: {} virhe abs.: {}".format(xmin, xmax,jako, erotus))
        
    testi = monte_carlo_integration(testifun1, xmin, xmax, blocks, iterations)
    erotus = np.abs(oikeaF1- testi[0])
    print("Monte carlo: F1({} , {}) blocks: {} iter : {} virhe abs.: {}".format(xmin, xmax,blocks, iterations, erotus))
    
    testi = monte_carlo_integration(testifun2, xmin, xmax, blocks, iterations)
    erotus = np.abs(oikeaF2- testi[0])
    print("Monte carlo: F2({} , {}) blocks: {} iter : {} virhe abs.: {}".format(xmin, xmax,blocks, iterations, erotus))
    
    testi = monte_carlo_integration(testifun3, xmin, xmax, blocks, iterations)
    erotus = np.abs(oikeaF3- testi[0])
    print("Monte carlo: F3({} , {}) blocks: {} iter : {} virhe abs.: {}".format(xmin, xmax,blocks, iterations, erotus))
    
        
def main():
    
    gradtest2D()
    gradtest3D()
    SGDtest()
    
    
        
if __name__ == "__main__":
    main()
   
   