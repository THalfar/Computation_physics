from numpy import *

"""
Finding the desired index from array, the array is equally spaced and uniform one. In linear_interp.py and spline_class.py function where was used. Here I use converting of numpy array to Python list with tolist() and find the intex by index()
"""

a = linspace(0,100,100)

rmax = 100
r_0 = 1e-5
dim = 100
r = zeros((dim,))
h = log(rmax/r_0+1)/(dim-1)
for i in range(1,dim):
    r[i] = r_0*(exp(i*h)-1)

"""
Printing every index
for val in r:
    print(r.tolist().index(val))

for val in a:
    print(a.tolist().index(val)) 
"""
"""
Print desired index
"""
print(r.tolist().index(r[21]),a.tolist().index(a[21]))
