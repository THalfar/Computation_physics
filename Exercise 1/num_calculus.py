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
   

def main():
    
    test_first_derivate(1.42, 0.0001)
    test_second_derivate(1.42, 0.0001)
    
    test_first_derivate(-1.42, 0.001)
    test_second_derivate(-1.42, 0.001)
    
if __name__ == "__main__":
    main()
   
   