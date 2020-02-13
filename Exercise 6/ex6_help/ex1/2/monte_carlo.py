import numpy as np

def monte_carlo_integration(fun,xmin,xmax,blocks,iters):
    """
    Monte Carlo integral
    :param fun: function to integrate
    :param xmin: range start
    :param xmax: range end
    :param blocks: number of blocks (resolution)
    :param iters: number of iterations (larger number will result in more accurate result)
    :return:
    """
    block_values = np.zeros((blocks,))
    L = xmax-xmin

    for block in range(blocks):
        for i in range(iters):
            x = xmin+np.random.rand()*L
            block_values[block] += fun(x)
        block_values[block] /= iters
    I = L*np.mean(block_values)
    dI = L*np.std(block_values)/np.sqrt(blocks)

    print(I)
    return I, dI

def func(x):
    """
    Test function, sin(x)
    :param x: x
    :return: sin(x)
    """
    return np.sin(x)

def main():
    """
    Main function
    :return: prints the test result, no return
    """
    I, dI = monte_carlo_integration(func, 0., np.pi/2, 10, 100)
    print(I, '+/-', 2*dI)

if __name__ == "__main__":
    main()