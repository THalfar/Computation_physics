import numpy as np
import matplotlib.pyplot as plt

""" FYS-4096 Computational Physics: Exercise 2 """
""" Author: Santeri Saariokari """

def root_search_linear(value, x):
    """
    Returns index i of x so that x[i] < value < x[i+1]
    or x[len(x)] if the value is larger than the last element of x
    """
    if x[0] - value > 1e-10 or x[-1] - value < 1e-10:
        print("Invalid value; %f is not in range of x"%(value))
        return len(x)
    num_points = len(x)
    value_range = x[-1] - x[0]

    # (value-x[0]) / value_range = "relative position in the area"
    # (num_points-1) = amount of dx
    # int works as floor
    start_index = int(((value-x[0]) / value_range) * (num_points-1))
    return start_index


def root_search_exp(value, x, r_0, h):
    """
    Returns index i of x so that x[i] < value < x[i+1]
    or x[len(x)] if the value is larger than the last element of x

    value = value to be searched for
    x = where "value" is searched from
    r_0 = smalles grid value
    h = grid parameter, specified by equation h = log(r_max / r_0 + 1) / (dim - 1)
    """
    if x[0] - value > 1e-10 or x[-1] - value < 1e-10:
        print("Invalid value; %f is not in range of x"%(value))
        return len(x)
    
    # We can simply solve equation r=r_0*(exp(i*h)-1) for i
    # to get i = log(r / r_0 + 1) / h
    start_index = int(np.log(value / r_0 + 1) / h)
    return start_index


def test_root_search_linear():
    print("=== Test for linear root search ===")
    x = np.linspace(0, 10, 30)
    val_1 = 1
    val_2 = 2.4
    val_3 = 9.9
    val_4 = 30

    index_1 = root_search_linear(val_1, x)
    index_2 = root_search_linear(val_2, x)
    index_3 = root_search_linear(val_3, x)
    index_4 = root_search_linear(val_4, x)

    print("x:\n", x, "\n index for %f = %d\n index for %f = %d\n index for %f = %d\n index for %f = %d"
              %(val_1, index_1, val_2, index_2, val_3, index_3, val_4, index_4))


def test_root_search_exp():
    print("=== Test for exponential root search ===")
    rmax = 100
    r_0 = 1e-5
    dim = 100
    h = np.log(rmax / r_0 + 1)/(dim - 1)

    x = np.zeros(dim)
    for i in range(1, dim):
        x[i] = r_0 * (np.exp(i*h)-1)

    val_1 = 1e-4
    val_2 = 0.0024
    val_3 = 0.99
    val_4 = 30
    index_1 = root_search_exp(val_1, x, r_0, h)
    index_2 = root_search_exp(val_2, x, r_0, h)
    index_3 = root_search_exp(val_3, x, r_0, h)
    index_4 = root_search_exp(val_4, x, r_0, h)

    print("x:\n", x, "\n index for %f = %d\n index for %f = %d\n index for %f = %d\n index for %f = %d"
              %(val_1, index_1, val_2, index_2, val_3, index_3, val_4, index_4))
    
    plt.figure()
    plt.plot(x)
    plt.xlabel('i')
    plt.ylabel('x[i]')
    plt.title('Exponential grid')
    plt.show()


def main():
    test_root_search_linear()
    test_root_search_exp()

if __name__ == "__main__":
    main()
