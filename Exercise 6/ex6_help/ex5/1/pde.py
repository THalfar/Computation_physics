"""
Problem 3 of exercise 5
Computational physics

Juha Teuho
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def jacobianUpdate(A, rho, eps0=1):
	"""
	Update the solution for pde using Jacobi method
	
	Args:
	    A (np.ndarray): 2D-matrix describing the current state
	    rho (np.ndarray): Initial distribution with stepsize term (h^2) included
	    eps0 (int, optional): permittivity of free space
	
	Returns:
	    np.ndarray: Updated solution
	"""
	A_new = 1.0*A
	for i in range(1,A.shape[0]-1):
		for j in range(1,A.shape[1]-1):
			A_new[i][j] = 1/4 * (A[i+1][j]+A_new[i-1][j]+A[i][j+1]+A[i][j-1] + rho[i][j]/eps0)
	return A_new

def gaussUpdate(A, rho, eps0=1):
	"""
	Update the solution for pde using Gauss method
	
	Args:
	    A (np.ndarray): 2D-matrix describing the current state
	    rho (np.ndarray): Initial distribution with stepsize term (h^2) included
	    eps0 (int, optional): permittivity of free space
	
	Returns:
	    np.ndarray: Updated solution
	"""
	A_new = 1.0*A
	for i in range(1,A.shape[0]-1):
		for j in range(1,A.shape[1]-1):
			A_new[i][j] = 1/4 * (A[i+1][j]+A_new[i-1][j]+A[i][j+1]+A_new[i][j-1] + rho[i][j]/eps0)
	return A_new

def sorUpdate(A, rho, eps0=1, omega=1.8):
	"""
	Update the solution for pde using SOR method
	
	Args:
	    A (np.ndarray): 2D-matrix describing the current state
	    rho (np.ndarray): Initial distribution with stepsize term (h^2) included
	    eps0 (int, optional): permittivity of free space
	    omega (float, optional): relaxation factor
	
	Returns:
	    np.ndarray: Updated solution
	"""
	A_new = 1.0*A
	for i in range(1,A.shape[0]-1):
		for j in range(1,A.shape[1]-1):
			A_new[i][j] = (1-omega)*A[i][j] + omega/4 * (A[i+1][j]+A_new[i-1][j]+A[i][j+1]+A_new[i][j-1] + rho[i][j]/eps0)
	return A_new


def iterate(A, rho, max_error):
	"""
	Iterate multiple steps using the SOR method
	
	Args:
		A (np.ndarray): Matrix to be solved
		rho (np.ndarray): Initial distribution with stepsize term (h^2) included
	    max_error (float): Maximum tolerated error
	
	Returns:
	    np.ndarray: Solution matrix
	"""
	A_new = A
	i = 0;
	while True:
		A = A_new
		A_new = sorUpdate(A_new, rho)
		if (np.amax(abs(A_new-A))) < max_error:
			break
	return A_new

def main():
	# Initialize the matrix with boundary conditions
	gridsize = 20
	A = np.zeros([gridsize, gridsize])
	rho = np.zeros([gridsize, gridsize])
	A[:, gridsize - 1] = 1

	# make grid
	x=np.linspace(0,1,gridsize)
	y=x
	[X,Y] = np.meshgrid(x,y)

	# find the solution and plot it
	Phi = iterate(A, rho, 0.00001)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.plot_wireframe(X,Y,Phi,rstride=1,cstride=1)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel(r'$\Phi$')
	plt.show()

if __name__=="__main__":
    main()
