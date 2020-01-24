import numpy as np
import matplotlib.pyplot as plt

def viiva(alku, loppu, n, q):
    
    if n<3:
        print("ei alle n<3")
        return
    
    qn = q/n
    
    viiva = np.zeros((n,3))
    viiva[0, :] = np.append(alku, qn) # add head
    # viiva[-1,:] = np.append(loppu,qn) # add tail
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
        # print(varaus)
        etaisyys = np.sqrt((varaus[0]-x)**2 + (varaus[1]-y)**2 )
        
        erotusvektoriX =  x- varaus[0] 
        Ex += varaus[-1] * np.divide( erotusvektoriX, etaisyys**3, out = np.zeros_like(etaisyys),where=etaisyys>0)
        
        erotusvektoriY = y - varaus[1] 
        Ey += varaus[-1] * np.divide( erotusvektoriY, etaisyys**3, out = np.zeros_like(etaisyys), where=etaisyys>0)
        
    return Ex, Ey
        
x = np.linspace(-3,3,20)
y = np.linspace(-3,3,20)
X,Y = np.meshgrid(x,y)

fig, ax = plt.subplots()

alku = np.array([0,0])
varaukset = ympyra(alku,np.pi,2*np.pi, 1.33, 1000,-1)
ax.plot(varaukset[:,0], varaukset[:,1], 'bo')

silmat = np.array([[-0.5,1,0.5],[0.5,1,0.5]])
varaukset = np.concatenate((varaukset, silmat), axis = 0)
ax.plot(varaukset[-2:,0], varaukset[-2:,1], 'ro')

Ex, Ey = test(X,Y, varaukset)
ax.quiver(x, y, Ex, Ey)

