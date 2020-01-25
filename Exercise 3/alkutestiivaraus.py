import numpy as np
import matplotlib.pyplot as plt

def viiva(alku, loppu, n, q):
    """
    makes a line of charge points

    Parameters
    ----------
    alku : np.array(1,2)
        starting point of line
    loppu : np.arra(1,2)
        ending point of line
    n : int
        number of charges along line
    q : float
        total charge in line

    Returns
    -------
    viiva : np.array(n,3)
            array of charges along line

    """
    
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
    """
    Makes an circle of charges

    Parameters
    ----------
    paikka : np.array(1,2)
        place of the circle center
    alkukulma : float
        starting angle of circle
    loppukulma : float
        end angle of circle
    r : float
        the diameter of circle
    n : int
        how many charge point at circle
    q : float
        total charge at circle

    Returns
    -------
    viiva : np.arra(n,3)
        array of charges along circle

    """
    
    qn = q/n    
    kulma = np.linspace(alkukulma,loppukulma,n)
    
    viiva = np.zeros((n,3))
    for i,aste in enumerate(kulma):
        lisays = paikka + r * np.array([np.cos(aste), np.sin(aste)])
        viiva[i,:] = np.append(lisays, qn)
    
    return viiva


def Epoints(X,Y, varaukset):
    """
    Calculates Electric field from point charges 

    Parameters
    ----------
    X : np.array
        meshrid x-coords
    Y : np.array
        meshridg y-coords
    varaukset : np.array(n,3) 
        array of charges, first indexs coords in xy, last charge amount

    Returns
    -------
    Ex : np.array
        meshgrid of E field streight in x-coord
    Ey : np.array
        meshgrid of E field streight in y-coord

    """
    
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    
    for varaus in varaukset:

        etaisyys = np.sqrt((varaus[0]-X)**2 + (varaus[1]-Y)**2 )
        
        erotusvektoriX =  X - varaus[0] 
        Ex += varaus[-1] * np.divide( erotusvektoriX, etaisyys**3, out = np.zeros_like(etaisyys),where=etaisyys>0)
        
        erotusvektoriY = Y - varaus[1] 
        Ey += varaus[-1] * np.divide( erotusvektoriY, etaisyys**3, out = np.zeros_like(etaisyys), where=etaisyys>0)
        
    return Ex, Ey
        
x = np.linspace(-4,4,50)
y = np.linspace(-4,4,50)
X,Y = np.meshgrid(x,y)

fig, ax = plt.subplots()

alku = np.array([0,0])
loppu = np.array([2,-1])
varaukset = ympyra(alku,np.pi,2*np.pi, 1.5, 500,-1)
# varaukset = viiva(alku, loppu, 100, 1)
ax.plot(varaukset[:,0], varaukset[:,1], 'bo')

# alku = np.array([-1,1])
# loppu = np.array([1,1])
# silmat = viiva(alku, loppu,100,-1)
silmat = np.array([[-0.5,1,0.5],[0.5,1,0.5]])
varaukset = np.concatenate((varaukset, silmat), axis = 0)
ax.plot(varaukset[-2:,0], varaukset[-2:,1], 'ro')
# ax.plot(varaukset[100:,0], varaukset[100:,1], 'bo')

Ex, Ey = Epoints(X,Y, varaukset)
# ax.quiver(x, y, Ex, Ey)
ax.streamplot(X,Y,Ex,Ey)
ax.set_aspect('equal')


