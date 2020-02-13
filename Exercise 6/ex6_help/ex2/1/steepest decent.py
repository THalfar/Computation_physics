import numpy as np
from gradient import gradient

def steepest_decent(fun,maxiters,minstep,a,h, start):
    """
    steepest_decent numerical function
    :param fun: function
    :param maxiters: number of items used if minstep not reached
    :param minstep: stops function if steps is smaller than this
    :param a: parameter
    :param h: dx or step size for gradient function
    :param start: point where you want to start looking at
    :return: coordinates for min
    """
    next_coord = start
    # looping through all the iters
    for i in range(maxiters):
        current_coord = next_coord
        # gamma based on lecture materials formula 31
        gamma = a/(len(gradient(fun,h ,current_coord))+1)
        # next coord based on lucture materials formula 32
        next_coord = current_coord-gamma*gradient(fun,h ,current_coord)
        # calculating step size
        step = np.linalg.norm(next_coord - current_coord)
        # check if step is too small
        if step < minstep:
            break
    return next_coord

def fun1(x):
    # 1d test function
    return x[0]**2+x[0]

def fun2(x):
    # 2d test function
    return x[0]*x[1]**2+1

def test_steepest_decent():
    """
    testing steepest_decent and printing results compared to real values
    """
    print("this should be [-0.5]: ", steepest_decent(fun1, 1000, 0.001, 1, 0.001, [3]))
    print("this should be [1,0]: ", steepest_decent(fun2, 1000, 0.001, 1, 0.001, [1, 0.5]))

def main():
    test_steepest_decent()

if __name__ == "__main__":
    main()
