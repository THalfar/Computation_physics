"""
Problem 4 from exercise 5
Computational Physics

Juha Teuho
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
			if not (isBoundaryPoint(i,j)):
				A_new[i][j] = 1/4 * (A[i+1][j]+A_new[i-1][j]+A[i][j+1]+A_new[i][j-1] + rho[i][j]/eps0)
	return A_new

def iterate(A, rho, N):
	"""
	Iterate multiple steps using the SOR method
	
	Args:
		A (np.ndarray): Matrix to be solved
		rho (np.ndarray): Initial distribution with stepsize term (h^2) included
	    max_error (float): Maximum tolerated error
	
	Returns:
	    np.ndarray: Solution matrix
	"""
	gridsize = N
	A_new = A
	i = 0;
	while i < 100:
		A_new = gaussUpdate(A_new, rho)
		i+=1
	return A_new

def isBoundaryPoint(x,y):
	"""
	Check if point (x,y) is a boundary point. This function could be made a lot
	better. I didn't have enough time to do that, though.
	
	Args:
	    x (int): x-coordinate
	    y (int): y-coordinate

	Returns:
	    boolean: True if point (x,y) is a boundary point.
	"""
	plate_1_x = 14
	plate_2_x = 26

	yn = np.linspace(-1,1,41)

	plates_y1 = np.where(yn == -0.5)
	plates_y2 = np.where(yn == 0.5) 

	if (x == plate_1_x and plates_y1[0][0] <= y <= plates_y2[0][0]):
		return True
	if (x == plate_2_x and plates_y1[0][0] <= y <= plates_y2[0][0]):
		return True
	return False



def main():
	# Make grid
	N=41
	x = np.linspace(-1,1,N)
	y = np.linspace(-1,1,N)
	rho = np.zeros([N, N])
	[X,Y] =np.meshgrid(x,y)

	# Define boundary conditions
	Psi = np.zeros([N,N])
	#plate_1_x = np.where(x == -0.3)  # From some reason these won't work
	#plate_2_x = np.where(x == 0.3)   # Why these don't give anything?
	# Need to define them manually for one grid size
	plate_1_x = 14
	plate_2_x = 26

	plates_y1 = np.where(y == -0.5) # these work for some reason
	plates_y2 = np.where(y == 0.5) 

	# Set boundary conditions for capacitor plates
	for i in range(plates_y1[0][0],plates_y2[0][0]+1):
		Psi[plate_1_x,i] = 1
		Psi[plate_2_x,i] = -1

	# Get the solution
	Psi = iterate(Psi,rho,N)

	# Plot solution
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.set_title('Electric potential')
	ax.set_xlabel(r'$x$')
	ax.set_ylabel(r'$y$')
	ax.set_zlabel(r'$\Psi$')
	ax.plot_wireframe(X,Y,Psi,rstride=1,cstride=1)

	# Make quiver plot for e-field
	E = np.array(np.gradient(Psi))
	Ex = E[0][:][:]
	Ey = E[1][:][:]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.quiver(x,y,Ex,Ey,width=0.007)
	ax.set_title('Electric field for two capacitor plates.')
	ax.set_xlabel(r'$x$')
	ax.set_ylabel(r'$y$')
	fig.tight_layout(pad=1)
	fig.savefig('quiverplot.pdf', dpi=200)
	plt.show()

if __name__=="__main__":
    main()