import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Regularly-spaced, coarse grid
n_data = 30
xmax, ymax = 2, 2
x = np.linspace(-xmax, xmax, n_data)
y = np.linspace(-ymax, ymax, n_data)
X, Y = np.meshgrid(x, y)
# Z = np.exp(-(2*X)**2 - (Y/2)**2)*np.sin(X)
# Z = np.sin(np.sqrt(X**2+Y**2))
Z = (X+Y) * np.exp(-np.sqrt(X**2+Y**2))
f = interp2d(x, y, Z, kind='cubic')

# Regularly-spaced, fine grid
n_int = 50
x2 = np.linspace(-xmax, xmax, n_int)
y2 = np.linspace(-ymax, ymax, n_int)
X2, Y2 = np.meshgrid(x2,y2)
Z2 = f(y2, x2)

fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d'})
ax[0].plot_wireframe(X, Y, Z, color='b')

ax[1].plot_wireframe(X2, Y2, Z2, color='b')
# for axes in ax:
#     axes.set_zlim(-0.2,1)
#     axes.set_axis_off()

fig.tight_layout()
plt.show()

n_int = 100
x3 = np.linspace(0,2/ np.sqrt(1.75),n_int)
y3 = np.sqrt(1.75) * x3
# x3.resize(1,100)
# y3.resize(1,100)
# points = np.concatenate((x3,y3), axis = 1)

oikeat = []
interpol = []
for i in range(n_int):
    oikea = (x3[i]+y3[i]) * np.exp(-np.sqrt(x3[i]**2 + y3[i]**2))
    oikeat.append(oikea)
    inter = f(x3[i],y3[i])
    interpol.append(inter)
    
n_int = np.arange(0,100)
plt.plot(n_int, oikeat, 'ro')
plt.plot(n_int, interpol, 'b-')

plt.show()