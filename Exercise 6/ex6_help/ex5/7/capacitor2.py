"""
Calculating the electric field around two capacitor plates in 2D.
"""
import numpy as np
import matplotlib.pyplot as plt



def solve_GS(points):
    
    # x and y indices for iteration, not including boundary points.
    x = np.linspace(1, points.shape[0] - 2, points.shape[0] - 2)
    y = np.linspace(1, points.shape[1] - 2, points.shape[1] - 2)
    
    max_diff = 1
    
    # Repeating until maximum absolute difference is under 10^-10
    while max_diff > 1e-10:
        # Make new array for values from this iteration.
        new_points = np.zeros_like(points)
        new_points[:,:] = points[:,:]
        # For every non-boundary x and y
        for i in x:
            for j in y:
                i = int(i)
                j = int(j)
                
                # Check if the point is in one of the plates, and if so, don't
                # change it
                if points[i, j] != 1 and points[i, j] != -1:
                    new_points[i, j] = 1/4*(points[i+1,j] + new_points[i-1, j] + points[i, j+1] + new_points[i, j-1])
        
        max_diff = np.max(np.abs(points - new_points))
        # Replace the old values with the new ones.
        points = new_points
        
    return points

points = np.zeros((20,20))
points[7,5:15] = 1
points[13,5:15] = -1


solution = solve_GS(points)

# Points for the plot.
x = np.linspace(-1,1,20)
y = np.linspace(-1,1,20)
X,Y = np.meshgrid(x, y)

gradient = np.gradient(solution)

fig, ax = plt.subplots()
q = ax.quiver(X,Y,-1*gradient[0],-1*gradient[1])
plt.plot([-0.3, -0.3],[0.5, -0.5])
plt.plot([0.3, 0.3],[0.5, -0.5], color="red")
plt.show()

