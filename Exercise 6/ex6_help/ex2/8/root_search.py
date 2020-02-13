""" 
-------- EXERCISE 2 - problem 4 ------------
----- FYS-4096 - Computational Physics -----

root_search.py finds the best starting points in a grid to start
root searhcing based on a probable starting point input

"""


import numpy as np
def linear_root_search(value, x):    
    """
    Finds the starting index for root searching
    """

    # if data in range
    if not (x[0] <= value <= x[-1]):
    
        print("Value {}  not in the given range".format(value))
        return False
        
    # Calculate length of the series N and its value range R
    
    N = len(x)    
    R = x[-1]-x[0]
    
    # Calculate starting index with floor function  and return it
    
    return int(np.floor((value-x[0])/R*(N-1)))
    
def exponential_root_search(value, x, r_0, h):
    """ 
    Finds the starting index for root searching
    """
    
    # if data in range
    if not (x[0] <= value <= x[-1]):
    
        print("Value {} not in the given range".format(value))
        return False
        
    # Calculate starting index with floor function  and return it
    return int(np.floor(1/h*np.log(value/r_0+1)))    

def test_lin_root_search():
    """
    tests the root search functions
    """
    x = np.linspace(0, 10, 20)
    r1 = linear_root_search(1.2, x) 
    r2 = linear_root_search(5.4, x)
    r3 = linear_root_search(15, x)
    print(x)
    print("1.1 index: {} and 7.7 index {}".format(r1,r2))

def test_exp_root_search():
    """
    tests the root search functions
    """
    
    r0 = 1e-3   
    rmax = 10   
    dim = 5
    h = np.log(rmax/r0+1)/(dim-1)    
    x = np.zeros(dim) 
       
    for i in range(dim):
        x[i] = r0*(np.exp(i*h)-1)
        r1 = exponential_root_search(1.2, x, r0, h)    
        r2 = exponential_root_search(5.4, x, r0, h)    
        r3 = exponential_root_search(100, x, r0, h)    
        print(x[i])
        print("1.1 index: {} and 7.7 index {}".format(r1,r2))
        
def main():
    test_lin_root_search()
    test_exp_root_search()
    
    
    

if __name__=="__main__":
    main()
