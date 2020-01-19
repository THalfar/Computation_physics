import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from num_calculus import monte_carlo_integration


# Intergoitavat funktiot
def eka(r): return r**2 * np.exp(-2*r)

def toka(x): return np.sin(x)/x

def kolmas(x): return np.exp(np.sin(x**3))

def testi(x): return x**2

def ir_int(fun, x0, tol, vali,points):
    """
    ir_int calculates indefinite integral x0 to inf

    Parameters
    ----------
    fun : function
        function what integrate
    x0 : double
        starting point of integration
    tol : double
        tolerance for which stop integrating
    vali : double
        one step length
    points : int
        amount of points in one integral step

    Returns
    -------
    tulos : double
        Integration value

    """

    tulos = 0        
    x1 = x0 + vali
    
    while True:    
        grid = np.linspace(x0, x1, points)
        y = fun(grid)
        I = integrate.simps(y, grid)
        tulos += I        
        if I < tol:
            break
        x0 = x1
        x1 += vali
        
    return tulos

def test_intEka():
    """
    Plot how error changes with tolerances to check if working

    Returns
    -------
    None.

    """
   
    true = 0.25  # from wolfram alpha 
    tolerances = np.geomspace(1e-6,1e-1,50)    
    integral_errors = []
    
    for tol in tolerances:                
        integral_errors.append(np.abs(ir_int(eka, 0, tol, 1, 100) - true))
        
    
    plt.plot(tolerances, integral_errors, 'ob', label = r' Int 0 to inf $f(r) = r^2 * exp(-2r) $ ')
    plt.xlabel("tol")
    plt.ylabel("Abs. error")    
    plt.xlim(tolerances[-1], tolerances[0])
    plt.title("Integral error")    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc = 0)
    plt.show()
    
def sig_int(fun, x0, x1, eps, points):
    """
    Integrates a function that have singularity at point x0

    Parameters
    ----------
    fun : function
        function that is going to be integrated
    x0 : double
        starting value 
    x1 : double
        integral end value
    eps : double
        epsilon i.e what is the minimum difference to singularity point x0
    points : int
        points amount used in integral

    Returns
    -------
    double
        Value of the integral

    """
    
    x = np.geomspace(x0+eps, x1, points)
    y = fun(x)
    return integrate.simps(y,x)

def test_intToka():
    """
    Plot the error in function of epsilon to check if working

    Returns
    -------
    None.

    """
    true = 0.946083070367183014941353313823179657812337954738111790471 # wolfram alpha
    epsilons = np.geomspace(1e-6, 1e-1, 50)
    integral_errors = []
    
    for eps in epsilons:
        integral_errors.append(np.abs(sig_int(toka, 0, 1, eps, 100)- true))
        
                 
    plt.plot(epsilons, integral_errors, 'ob', label = r'Int from 0 to 1 $f(r) = r^2 * exp(-2r) $ ')
    plt.xlabel("epsilon")
    plt.ylabel("Abs. error")    
    plt.xlim(epsilons[-1], epsilons[0])
    plt.title("Integral error")    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc = 0)
    plt.show()
        
def swing_int(fun, x0, x1, blocks, points):
    """
    use monte carlo to integrate a wildy swinging function at endpoint
    # vähän kiire, tämä toimii ja käytetään nyt valmiita
    Parameters
    ----------
    fun : functipn
        function what to integrate
    x0 : double
        start point of integration
    x1 : double
        end point of integration
    blocks : int
        amount of blocks
    points : int
        how many points are calculated per block

    Returns
    -------
    double
        value of intergral

    """
    
    return monte_carlo_integration(fun, x0, x1, blocks, points)

def test_intKolmas():
    
    #wolfram alpha
    true = 6.647272079953789849569443469671631160297360492445608431
    blocks = np.arange(1,1000,100)    
    integral_errors = []
    
    for block in blocks:
        integral_errors.append(np.abs(monte_carlo_integration(kolmas, 0, 5, block, 1000)[0]- true))
        
    plt.plot(blocks*1000, integral_errors, 'ob', label = r'Int from 0 to 5 $f(x) = exp(sin(x^3)) $ ')
    plt.xlabel("number of points")
    plt.ylabel("Abs. error")    
    # plt.xlim(blocks[-1], blocks[0])
    plt.title("Integral error")    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc = 0)
    plt.show()
 
def monte_carlo_box(fun, starts, ends, points):
    """
    Integrates N-dimensional monte carlo box
    # TODO tämä on yksi boksi, jaa moneen boksiin ja käytä tätä kuha aikaa on

    Parameters
    ----------
    fun : function
        function that takes a vector in
    starts : numpy array 
        start points in array form
    ends : numpy array
        end points in array form
    points : int
        in how many points calculate the value of function

    Returns
    -------
    double
        value of integral

    """
    
    V = np.prod(np.abs(starts-ends))
    
    summation = 0
    
    for i in range(points):
        
        random_vector = []
        
        for j in range(len(ends)):
            
            dice = np.random.random() * (ends[j] - starts[j]) + starts[j]
            random_vector.append(dice)
         
        summation += fun(random_vector)
        
    summation /= points
    
    return summation * V
            
# 2c tehtävän funktio    
def neljas(vektori):
    x = vektori[0]
    y = vektori[1]
    return x * np.exp(-np.sqrt(x**2 + y**2))        

# tehtävän 2c tarkastus
def test_intNeljas():
    
    oikea = 1.57347 # wolfram alphasta
    starts = np.array([0, -2])
    ends = np.array([2, 2])    
    points = np.geomspace(10,1e6,42,dtype = int)
    integral_errors = []
    
    for point in points:
        integral_errors.append(np.abs(monte_carlo_box(neljas, starts, ends, point) - oikea))
        
    plt.plot(points, integral_errors, 'ob')
    plt.xlabel("number of points")
    plt.ylabel("Abs. error")    
    # plt.xlim(blocks[-1], blocks[0])
    plt.title("Integral error of 2c")    
    plt.xscale('log')
    plt.yscale('log')    
    plt.show()
    
def monte_carlo_3D_ball(r, ra, rb, points, iterations):
    """
    Integrates N-dimensional monte carlo ball
        
    Parameters
    ----------
    fun : function
        function that takes a vector in
    starts : numpy array 
        start points in array form
    ends : numpy array
        end points in array form
    points : int
        in how many points calculate the value of function
    
    Returns
    -------
    double
        value of integral
    
    """
    
    V = (4/3) * np.pi * r**3
    integraalit = []
    
    for j in range(iterations):
        summation = 0
               
        for i in range(points):
            
            while True:
                random_vector = -2*r*np.random.random(3) + r
                if np.linalg.norm(random_vector) < r:
                    break                                                       
                         
            atrain = np.exp(-1*np.linalg.norm(random_vector-ra)) / np.sqrt(np.pi)
            jaettava = atrain**2
            jakaja = np.linalg.norm(random_vector - rb)
            tulos = jaettava / jakaja
            summation += tulos
            
        summation /= points        
        integraalit.append(summation) 
        
    return np.mean(integraalit)*V
 

def test_intViides():
    
    r = 4
              
    for i in range(5):
        
        ra = -2*np.random.rand(3)+2
        rb = -2*np.random.rand(3)+2                   
        R = np.linalg.norm(ra-rb)        
        oikea = (1- (1+R)*np.exp(-2*R))/R
        
        tulos = monte_carlo_3D_ball(r, ra, rb, int(1e3), 10)
        print("ra: {} rb: {} R: {}".format(ra,rb,R))
        print("Oikea : {}".format(oikea))
        print("Carlo : {}".format(tulos) )        
        print("Virhe : {}".format(np.abs(oikea-tulos)))
        print("Virhe%: {:.2%}".format(np.abs(oikea-tulos)/oikea))        
        print("")
        
        
    
def main():

    # test_intEka()
    # test_intToka()
    # test_intKolmas()
    # test_intNeljas()
    test_intViides()
    
   
if __name__ == "__main__":
    main()
   

