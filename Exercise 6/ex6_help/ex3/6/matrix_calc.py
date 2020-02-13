"""
Matrix calculation functions for course "Computational physics"

author: Juha Teuho
"""

import numpy as np

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def eigenvalue(A,x):
	""" Calculate the eigenvalue for matrix A when eigenvector x is known
	
	Args:
	    A (ndarray): Matrix
	    x (ndarray): Eigenvector
	
	Returns:
	    float: Eigenvalue
	"""

	Ax = np.dot(A,x)
	return np.dot(x,Ax)

def largest_eig(A, precision = 0.00001):
    """ Calculate the largest eigenvalue and the corresponding eigenvector of
    matrix A using the power method.
    
    Args:
        A (ndarray): Matrix
        precision (float, optional): Maximum error for eigenvalue
    
    Returns:
        eig_new (float): Largest eigenvalue of matrix A
        x_new (ndarray): Eigenvector corresponding to the largest eigenvalue
        n_iterations (int): Number of performed iterations
    """

    # Start with a random guess for eigenvector (size same as number of columns
    # in A.
    x_n = np.random.rand(A.shape[1])
    eig_n = eigenvalue(A, x_n)
    n_iterations = 0

    # Iterate new eigenvalues and -vectors until desired precision is obtained.
    while True:
        n_iterations += 1

        # make a new approximation for the eigenvector
        Ax = A.dot(x_n)
        x_n1 = Ax / np.linalg.norm(Ax)

        # make a new approximation for the eigenvalue based on new eigenvector
        eig_n1 = eigenvalue(A, x_n1)
        if np.abs(eig_n - eig_n1) < precision:
            break

        x_n = x_n1
        eig_n = eig_n1

    return eig_n1, x_n1, n_iterations

def random_test_matrix(rows=2,columns=2):
	return np.random.rand(rows,columns)

def test_largest_eig():
	test_matrix = random_test_matrix()
	eig = max(np.linalg.eig(test_matrix)[0])
	my_eig, v, n_iterations = largest_eig(test_matrix, 10**-4)

	print("Largest eigenvalue for matrix")
	matprint(test_matrix)
	print("Value given by numpy.linalg.eig:",eig)
	print("Value given by largest_eig:", my_eig)
	print("approximation took",n_iterations,"iterations.")

def main():
    # perform the tests
    test_largest_eig()

if __name__=="__main__":main()
