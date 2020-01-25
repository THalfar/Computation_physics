import numpy as np
import matplotlib.pyplot as plt

def viiva(alku, loppu, n, q):
    
    if n<3:
        print("ei alle n<3")
        return
    
    qn = q/n
    
    viiva = np.zeros((n,3))
    viiva[0, :] = np.append(alku, qn) # add head
    erotus = (loppu - alku) / (n-1)
    
    for i in range(1,n):
        viiva[i, :] = np.append(alku+erotus*i, qn)
    
    return viiva
        

def ympyra(paikka, alkukulma, loppukulma, r, n, q):
    
    qn = q/n    
    kulma = np.linspace(alkukulma,loppukulma,n)
    
    viiva = np.zeros((n,3))
    for i,aste in enumerate(kulma):
        lisays = paikka + r * np.array([np.cos(aste), np.sin(aste)])
        viiva[i,:] = np.append(lisays, qn)
    
    return viiva


def test(x,y, varaukset):
    
    Ex = np.zeros_like(x)
    Ey = np.zeros_like(y)
    
    for varaus in varaukset:

        etaisyys = np.sqrt((varaus[0]-x)**2 + (varaus[1]-y)**2 )
        
        erotusvektoriX =  x - varaus[0] 
        Ex += varaus[-1] * np.divide( erotusvektoriX, etaisyys**3, out = np.zeros_like(etaisyys),where=etaisyys>0)
        
        erotusvektoriY = y - varaus[1] 
        Ey += varaus[-1] * np.divide( erotusvektoriY, etaisyys**3, out = np.zeros_like(etaisyys), where=etaisyys>0)
        
    return Ex, Ey
        
x = np.linspace(-4,4,100)
y = np.linspace(-4,4,100)
X,Y = np.meshgrid(x,y)

fig, ax = plt.subplots()

alku = np.array([0,0])
loppu = np.array([2,-1])
varaukset = ympyra(alku,np.pi,2*np.pi, 1.5, 1000,-1)
# varaukset = viiva(alku, loppu, 100, 1)
ax.plot(varaukset[:,0], varaukset[:,1], 'bo')

# alku = np.array([-1,1])
# loppu = np.array([1,1])
# silmat = viiva(alku, loppu,100,-1)
silmat = np.array([[-0.5,1,0.5],[0.5,1,0.5]])
varaukset = np.concatenate((varaukset, silmat), axis = 0)
ax.plot(varaukset[-2:,0], varaukset[-2:,1], 'ro')
# ax.plot(varaukset[100:,0], varaukset[100:,1], 'bo')

Ex, Ey = test(X,Y, varaukset)
# ax.quiver(x, y, Ex, Ey)
ax.streamplot(X,Y,Ex,Ey)
ax.set_aspect('equal')

testi = (1 - 1/(3))*1/2


