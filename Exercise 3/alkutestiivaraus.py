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
 
    
    


# q = np.array([[0.5,5.5,1e-5], [1.1,2.1,0]])

def line(l, n, q):
    
    qn = q/n
    x = np.linspace(-l/2, l/2, n).reshape((n,1))
    y = np.zeros((n,1))
    q = np.ones((n,1))*qn 
    ulos = np.concatenate((x,y,q), axis = 1)
    return ulos
    

def ympyra(n,q):
    
    qn = q/n
    x = np.linspace(np.pi, 2*np.pi,n).reshape((n,1))
    y = np.sin(x)
    q = np.ones((n,1))*qn
    ulos = np.concatenate((x,y,q), axis = 1)
    return ulos
    


def test(x,y, varaukset):
    
    Ex = np.zeros_like(x)
    Ey = np.zeros_like(y)
    
    for varaus in varaukset:
        # print(varaus)
        etaisyys = np.sqrt((varaus[0]-x)**2 + (varaus[1]-y)**2 )
        
        erotusvektoriX =  x- varaus[0] 
        Ex += varaus[-1] * np.divide( erotusvektoriX, etaisyys**3, out = np.zeros_like(etaisyys),where=etaisyys!=0)
        
        erotusvektoriY = y - varaus[1] 
        Ey += varaus[-1] * np.divide( erotusvektoriY, etaisyys**3, out = np.zeros_like(etaisyys), where=etaisyys!=0)
        
    return Ex, Ey
        

# varaukset = line(1,1000,1e-6)

x = np.linspace(2,8,20)
y = np.linspace(-2,2,20)
X,Y = np.meshgrid(x,y)

varaukset = ympyra(100,-1)

fig, ax = plt.subplots()
ax.plot(varaukset[:,0], varaukset[:,1], 'bo')

uusia = np.array([[4,1,0.5], [6,1,0.5]])
varaukset = np.concatenate((varaukset,uusia), axis = 0)
ax.plot([4,6], [1, 1], 'ro')
Ex, Ey = test(X,Y, varaukset)
q = ax.quiver(x, y, Ex, Ey)

