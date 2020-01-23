import numpy as np
from scipy.integrate import simps, dblquad
import matplotlib.pyplot as plt

def fun(x,y): return (x+y)*np.exp(-np.sqrt(x**2 + y**2))

oikea = dblquad(fun, 0, 2, -2, 2)
print(oikea[0])

kokolo = np.geomspace(2,1000, 42, dtype = int)

virhe = []

for i in kokolo:
    x = np.linspace(0,2, 2*i)
    y = np.linspace(-2,2,i, endpoint = False)

    # y = np.concatenate((yminus, yplus), axis = 0)
    
    X,Y = np.meshgrid(x,y)
    Z = fun(X,Y)
    x_int = simps(Z, x)
    koko = simps(x_int, y)

    virhe.append(np.abs(koko-oikea[0]))
    print("Koko: {}, virhe: {}".format(i, np.abs(koko-oikea[0])))
    
plt.plot(kokolo, virhe, 'or')
plt.xscale('log')
plt.yscale('log')

