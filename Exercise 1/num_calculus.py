import numpy as np
"""
Calculates first derivate 

Param:
function  -  Function that shall be derivated
x         -  Where derivate
dx        -  Difference step in derivate calculation  

Return:
Value of first derivate
"""
def first_derivate(function, x, dx):
    
    jakaja = 2*dx
    jaettava = function(x+dx) - function(x-dx)
    
    return jaettava / jakaja

"""
Calculates second derivate 

Param:
function  -  Function that shall be derivated
x         -  Where derivate
dx        -  Difference step in derivate calculation  

Return: 
Value of second derivate    
"""
def second_derivate(function, x, dx):
    
    jakaja = dx**2
    jaettava = function(x+dx) + function(x-dx) - 2*function(x)
    
    return jaettava / jakaja

"""
Calculates a numerical integral of the trapezoid integral

Param:
x       -  a uniformly spaced x-values 
function -  function thats going to be integrated

Return:
Trapezoid integral of function
"""
def trapezoid_int(x, function):
    
    # calculate point interval, assuming that x has at least 2 points    
    h = x[1] - x[0]
        
    summa = 0
    
    # Loops through points x and add to the summa the block values
    for i in range(0, len(x)-1):
        
        summa += (function(x[i]) + function(x[i+1]))*h
        
    return summa * 0.5
    
    
def riemann_sum(x, function):
    
    # calculate point interval, assuming that x has at least 2 points    
    h = x[1] - x[0]
        
    summa = 0
    
    # Loops through points x and add to the summa the block values
    for i in range(0, len(x)-1):
        
        summa += (function(x[i+1]))*h
        
    return summa
    
def simpson_int(x, function):
    
    # calculate point interval, assuming that x has at least 2 points    
    h = x[1] - x[0]

    summa = 0
    
    # Is grid x even or odd
    if len(x)%2 == 0:
        pariton = False
    else:
        pariton = True
        
    askelia = int( (len(x) -1) / 2 - 1 )
    

    
    for i in range(askelia):
        # print(function(x[2*i]))
        # print(4*function(x[2*i+1]))
        # print(function(x[2*i+2]))
        summa += function(x[2*i]) + 4*function(x[2*i+1]) + function(x[2*i+2])

        
    return summa * h/3
        


"""
These are example functions for testing
"""
# f(x) = x^2 + 3x
def testifun1(x): return x**2 + 3*x

# f(x) = 2x^3+xˆ2+42
def testifun2(x): return 2* x**3 + x**2 + 42

# f(x) = exp(2x) + sin(x)
def testifun3(x): return np.exp(2*x) + np.sin(x)
    
    
"""
Testing first derivate with three functions
Param:
x   -  point where derivate
dx  -  difference step in derivate

Return:
None
"""
def test_first_derivate(x, dx):
    
    # f1(x) = x^2 + 3x -> 2x + 3
   oikea =  2*x+3
   testi = first_derivate(testifun1, x, dx)
   erotus = np.abs(oikea-testi)
   print("Erotus f1'  oikeaan tulokseen oli {}".format(erotus))
   
   # f2(x) = 2x^3+xˆ2+42 -> 6xˆ2 + 2x   
   oikea =  6*x**2 + 2*x
   testi = first_derivate(testifun2, x, dx)
   erotus = np.abs(oikea-testi)
   print("Erotus f2'  oikeaan tulokseen oli {}".format(erotus))
   
    # f3(x) = exp(2x) + sin(x) -> 2*exp(2x) - cos(x)
   oikea = 2* np.exp(2*x) + np.cos(x)
   testi = first_derivate(testifun3, x, dx)
   erotus = np.abs(oikea-testi)
   print("Erotus f3' oikeaan tulokseen oli {}".format(erotus))

"""
Testing second derivate with three function
Param:
x   -  point where derivate
dx  -  difference step in derivate

Return:
None
"""
def test_second_derivate(x, dx):
    
    # f1(x) = x^2 + 3x -> 2
    oikea = 2
    testi = second_derivate(testifun1, x, dx)
    erotus = np.abs(oikea-testi)
    print("Erotus f1'' oikeaan tulokseen oli {}".format(erotus))
    
    # f2(x) = 2x^3+xˆ2+42 -> 12x + 2  
    oikea = 12*x+2
    testi = second_derivate(testifun2, x, dx)
    erotus = np.abs(oikea-testi)
    print("Erotus f2'' oikeaan tulokseen oli {}".format(erotus))
    
    # f3(x) = exp(2x) + sin(x) -> 2*exp(2x) - sin(x)
    oikea = 4*np.exp(2*x) - np.sin(x)
    testi = second_derivate(testifun3, x, dx)
    erotus = np.abs(oikea-testi)
    print("Erotus f3'' oikeaan tulokseen oli {}".format(erotus))
   

def test_integral():
        
    x = np.linspace(0,1, 123)
    
    # f1(x) = x^2 + 3x -> 1/3 * x^3 + 3/2 * x^2
    oikea = ( 1/3 * x[-1]**3 + 3/2 * x[-1]**2 ) -  ( 1/3 * x[0]**3 + 3/2 * x[0]**2 ) 
    testi = riemann_sum(x, testifun1)
    erotus = np.abs(oikea-testi)
    print("Erotus F1 oikeaan tulokseen riemannin summalla oli {}".format(erotus))
        
    # f2(x) = 2x^3+xˆ2+42 -> 1/2 * x^4 + 1/3 x^3 + 42x 
    oikea = 1/2 + 1/3 + 42
    testi = riemann_sum(x, testifun2)
    erotus = np.abs(oikea-testi)
    print("Erotus F2 oikeaan tulokseen riemannin summalla oli {}".format(erotus))
    
    # f3(x) = exp(2x) + sin(x) -> 1/2 exp(2x) - cos(x)
    oikea = ( 1/2 * np.exp(2) - np.cos(1) ) - ( 1/2 * np.exp(0) - np.cos(0) )
    testi = riemann_sum(x, testifun3)
    erotus = np.abs(oikea-testi)
    print("Erotus F3 oikeaan tulokseen riemannin summalla oli {}".format(erotus))
    
    # f1(x) = x^2 + 3x -> 1/3 * x^3 + 3/2 * x^2
    oikea = ( 1/3 * x[-1]**3 + 3/2 * x[-1]**2 ) -  ( 1/3 * x[0]**3 + 3/2 * x[0]**2 ) 
    testi = trapezoid_int(x, testifun1)
    erotus = np.abs(oikea-testi)
    print("Erotus F1 oikeaan tulokseen trapedoizilla oli {}".format(erotus))
        
    # f2(x) = 2x^3+xˆ2+42 -> 1/2 * x^4 + 1/3 x^3 + 42x 
    oikea = 1/2 + 1/3 + 42
    testi = trapezoid_int(x, testifun2)
    erotus = np.abs(oikea-testi)
    print("Erotus F2 oikeaan tulokseen trapedoizilla oli {}".format(erotus))
    
    # f3(x) = exp(2x) + sin(x) -> 1/2 exp(2x) - cos(x)
    oikea = ( 1/2 * np.exp(2) - np.cos(1) ) - ( 1/2 * np.exp(0) - np.cos(0) )
    testi = trapezoid_int(x, testifun3)
    erotus = np.abs(oikea-testi)
    print("Erotus F3 oikeaan tulokseen trapedoizilla oli {}".format(erotus))
    
    
    x = np.linspace(0,1, 123)
    
    # f1(x) = x^2 + 3x -> 1/3 * x^3 + 3/2 * x^2
    oikea = ( 1/3 * x[-1]**3 + 3/2 * x[-1]**2 ) -  ( 1/3 * x[0]**3 + 3/2 * x[0]**2 ) 
    testi = simpson_int(x, testifun1)
    erotus = np.abs(oikea-testi)
    print("Erotus F1 oikeaan tulokseen simpsonilla oli {}".format(erotus))
        
    # # f3(x) = exp(2x) + sin(x) -> 1/2 exp(2x) - cos(x)
    # oikea = ( 1/2 * np.exp(x[-1]) - np.cos(x[-1]) ) - ( 1/2 * np.exp(x[0]) - np.cos(x[0]) )
    # testi = simpson_int(x, testifun3)
    # erotus = np.abs(oikea-testi)
    # print("Erotus F3 oikeaan tulokseen simpsonilla oli {}".format(erotus))
        
    
        
    
def main():
    
    test_first_derivate(1.42, 0.0001)
    test_second_derivate(1.42, 0.0001)
    
    test_first_derivate(-1.42, 0.001)
    test_second_derivate(-1.42, 0.001)
    
    test_integral()
    
if __name__ == "__main__":
    main()
   
   