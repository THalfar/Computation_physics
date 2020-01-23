import numpy as np
import matplotlib.pyplot as plt

# varaus = np.array([0,0,1], dtype = float, ndmin= 2)
# paikat = np.array([[1,1], [-1, -1]], dtype = float)

# testi = varaus[:, :-1] -paikat
# r_norm = np.linalg.norm(testi, axis = 1)

# voimax = np.zeros_like(paikat, dtype = float)

# for i, paikka in enumerate(paikat):
    
#     for qpaikka in varaus:
                    
#         voimax[i,0] += qpaikka[-1] * (qpaikka[:-1] - paikka)[0] / np.linalg.norm(paikka - qpaikka[:-1])**3
#         voimax[i,1] += qpaikka[-1] * (qpaikka[:-1] - paikka)[1] / np.linalg.norm(paikka - qpaikka[:-1])**3   
 
    
    

x = np.linspace(-2,2,20)
y = np.linspace(-1,1,20)
X,Y = np.meshgrid(x,y)

# q = np.array([[0.5,5.5,1e-5], [1.1,2.1,0]])

def line(l, n, q):
    
    qn = q/n
    y = np.linspace(-l/2, l/2, n).reshape((n,1))
    x = np.zeros((n,1))
    q = np.ones((n,1))*qn
    
   
    
    
    
    ulos = np.concatenate((x,y,q), axis = 1)
    return ulos
    
    


def test(x,y):
    
    Ex = np.zeros_like(x)
    Ey = np.zeros_like(y)
    
    for varaus in varaukset:
        # print(varaus)
        etaisyys = np.sqrt((x-varaus[0])**2 + (y-varaus[1])**2 )
        
        erotusvektoriX =  varaus[0] - x
        Ex += varaus[-1] * np.divide( erotusvektoriX, etaisyys**3, out = np.zeros_like(etaisyys),where=etaisyys!=0)
        
        erotusvektoriY = varaus[1] - y
        Ey += varaus[-1] * np.divide( erotusvektoriY, etaisyys**3, out = np.zeros_like(etaisyys), where=etaisyys!=0)
        
    return Ex, Ey
        
varaukset = line(1,1000,1)
Ex, Ey = test(X,Y)

fig, ax = plt.subplots()
q = ax.quiver(x, y, Ex, Ey)