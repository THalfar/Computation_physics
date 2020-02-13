import numpy as np

def root_search(x, fun):
    """
    function for root search, simply loops through all the elements of given x and check
    if function value has different sign for elements next to each other
    :param x: x coordinates
    :param fun: function
    :return: arrya of roots
    """
    roots = []
    for i in range(len(x)-1):
        if (fun(x[i])<=0 and fun(x[i+1])>0) or (fun(x[i])>=0 and fun(x[i+1])<0):
            roots.append(x[i])
    return roots

def fun1(x):
    """
    sample function
    :param x: x coordinate
    :return: function value
    """
    return x**2-1

def generate_grid(r_0, r_max, dim):
    # generates grid based on example on exercise sheet
    h = np.log(r_max/r_0+1)/(dim-1)
    r = np.zeros(dim)
    r[0] = 0.
    for i in range(1, dim):
        r[i] = r_0 * (np.exp(i * h) - 1)
    return r

def test_root_search():
    """
    testing root search with few examples
    """
    x = np.linspace(-2,2,100)
    print("this should be [-1,1]: ",root_search(x,fun1))
    xx = generate_grid(100, 2, 100)
    print("this should be [1]: ",root_search(xx,fun1))

def main():
    test_root_search()

if __name__ == "__main__":
    main()