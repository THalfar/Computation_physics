import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


def viiva(alku, loppu, n, q = None):
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

    if q != None:    
        qn = q/n
        viiva = np.zeros((n,3))
        viiva[0, :] = np.append(alku, qn) # add head
        erotus = (loppu - alku) / (n-1)
        
        for i in range(1,n):
            viiva[i, :] = np.append(alku+erotus*i, qn)            
            
    else:
        viiva = np.zeros((n,2))
        viiva[0, :] = alku # add head
        erotus = (loppu - alku) / (n-1)
        
        for i in range(1,n):
            viiva[i, :] = alku+erotus*i  
        
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
    
    Ex = np.zeros_like(X) # Initialize E x-coords
    Ey = np.zeros_like(Y) # Initialize E y-coords
    # iterate over all charges and calculate what electric field they make to Ex, Ey
    for varaus in varaukset:

        etaisyys = np.sqrt((varaus[0]-X)**2 + (varaus[1]-Y)**2 ) # length to charge
        
        erotusvektoriX =  X - varaus[0] # if divided by zero add value 0 to get away with NaN case
        Ex += varaus[-1] * np.divide( erotusvektoriX, etaisyys**3, out = np.zeros_like(etaisyys),where=etaisyys>0)
        
        erotusvektoriY = Y - varaus[1] 
        Ey += varaus[-1] * np.divide( erotusvektoriY, etaisyys**3, out = np.zeros_like(etaisyys), where=etaisyys>0)
        
    return Ex, Ey

# Test the Epoints method using a happy face
def face():     
    x = np.linspace(-5,5,50)
    y = np.linspace(-5,5,50)
    X,Y = np.meshgrid(x,y)
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    alku = np.array([0,0])    
    varaukset = ympyra(alku,np.pi,2*np.pi, 1.5, 300,-1)    
    ax.plot(varaukset[:,0], varaukset[:,1], 'bo')
    
    alku = np.array([-0.5,1])    
    silmat = ympyra(alku,0,2*np.pi, 0.3, 300,1)    
    varaukset = np.concatenate((varaukset, silmat), axis = 0)
    ax.plot(varaukset[-300:,0], varaukset[-300:,1], 'ro')
    
    alku = np.array([0.5,1])    
    silmat = ympyra(alku,0,2*np.pi, 0.3, 300,1)    
    varaukset = np.concatenate((varaukset, silmat), axis = 0)
    ax.plot(varaukset[-300:,0], varaukset[-300:,1], 'ro')
    
    alku = np.array([-1,2])
    loppu = np.array([1,2])
    lisaviiva = viiva(alku,loppu, 100, -1)
    varaukset = np.concatenate((varaukset, lisaviiva), axis = 0)
    ax.plot(varaukset[-100:,0], varaukset[-100:,1], 'bo')
        
    Ex, Ey = Epoints(X,Y, varaukset)
    ax.streamplot(X,Y,Ex,Ey)
    ax.set_aspect('equal')
    plt.title("Face using streamplot and pointlines")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    
  
def Esimpsline(alku, loppu, n, Q, X, Y, Ex, Ey):
    """
    Calculates Electric field of a continous line using simpson integral

    Parameters
    ----------
    alku : np.array(2,1)
        starting point
    loppu : np.array(2,1)
        ending point
    n : int
        number of simps integral points for this line
    Q : float
        total charge in this line
    X : np.array(size,size)
        meshgrid of X points value
    Y : np.array(size,size)
        meshgrid of y points value
    Ex : np.array(size,size)
        field streight meshgrid Ex coords
    Ey : np.array(size,size)
        field streight meshgrid Ey coords

    Returns
    -------
    Ex : np.array(size,size)
        Ex meshgrid with line charges added
    Ey : np.array(size,size)
        Ey meshgrid with line charges added

    """
    
    suora = viiva(alku, loppu, n)
    pituus = np.linalg.norm(alku-loppu)
    varaustiheys = Q / pituus
    
    for i in range(Ex.shape[0]):
        
        for j in range(Ex.shape[1]):
            
            simarvotX = []      
            simarvotY = []
            
    
            for arvo in suora:
                # print(arvo)
        
                simarvotX.append((X[i,j] - arvo[0]) / ( (X[i,j]-arvo[0])**2 + (Y[i,j]-arvo[1])**2 )**(3/2))                    
                simarvotY.append((Y[i,j] - arvo[1]) / ( (X[i,j]-arvo[0])**2 + (Y[i,j]-arvo[1])**2 )**(3/2))
        
            Ex[i,j] += simps(simarvotX) * varaustiheys
            Ey[i,j] += simps(simarvotY) * varaustiheys
    
    return Ex, Ey
    
    
    
    
def Elinerod(L, n, Q, x, y):
    """
    Calculates electric field of a uniformly charged rod at x-axis
    with length L centered at origin

    Parameters
    ----------
    L : float
        Lentgh of rod
    n : int
        grid spacing over to integrate
    Q : float
        total charge of rod
    x : float
        x-coord where to calculate E field
    y : float
        y-coord where to calculate E field

    Returns
    -------
    np.array
        E field strength E[0] x-axis, E[1] y-axis

    """
    eps = 8.8541878128e-12 # vacuum permittivy
    E = np.zeros(2) # Ex and Ey in array    
    lam = Q/L # value of lambda line charge per length
    x_grid = np.linspace(-L/2, L/2, n)
    
    x_values = (x -x_grid ) / ( (x - x_grid )**2 + y**2 )**(3/2)
    E[0] = simps(x_values, x_grid) * lam/(4*np.pi*eps)
    
    y_values = y /  ( (x_grid - x)**2 + y**2 )**(3/2) 
    E[1] = simps(y_values, x_grid)  * lam/(4*np.pi*eps)
    
    return E
    
# test the line integral method
def testRod(d, L ,Q, n):
    testi = Elinerod(L, n, Q, L/2+d,0) # test value        
    eps = 8.8541878128e-12 # vacuum permittivy
    real =(Q/L) * (1/(4*np.pi*eps )) * (1/d - 1/(d+L)) # analytical value
    print("Line integral gives at x-direction with {} points: {}".format(n, testi[0]))
    print("Analytical solution gives at x-direction: {}".format(real))
    print("Error: {:.6%}".format(np.abs(testi[0] - real)/np.abs(real)))
    
# test the point line with a bunch of lines    
def linestest():     
    x = np.linspace(-4,4,50)
    y = np.linspace(-4,4,50)
    X,Y = np.meshgrid(x,y)
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    alku = np.array([0,0])
    loppu = np.array([2,-1])
    varaukset = viiva(alku,loppu,100,1)
    ax.plot(varaukset[:,0], varaukset[:,1], 'ro')
    
    alku = np.array([0,0])
    loppu = np.array([-1,-1])
    lisaviiva = viiva(alku,loppu, 100, -1)
    varaukset = np.concatenate((varaukset, lisaviiva), axis = 0)
    ax.plot(varaukset[-100:,0], varaukset[-100:,1], 'bo')
    
    alku = np.array([1,1])
    loppu = np.array([3,2])
    lisaviiva = viiva(alku,loppu, 100, -1)
    varaukset = np.concatenate((varaukset, lisaviiva), axis = 0)
    ax.plot(varaukset[-100:,0], varaukset[-100:,1], 'bo')
    
    alku = np.array([-2,2])
    loppu = np.array([0,3])
    lisaviiva = viiva(alku,loppu, 100, 2)
    varaukset = np.concatenate((varaukset, lisaviiva), axis = 0)
    ax.plot(varaukset[-100:,0], varaukset[-100:,1], 'ro')
    
    alku = np.array([-3,-3])
    loppu = np.array([3,-3])
    lisaviiva = viiva(alku,loppu, 300, -1)
    varaukset = np.concatenate((varaukset, lisaviiva), axis = 0)
    ax.plot(varaukset[-300:,0], varaukset[-300:,1], 'bo')
    
    
    
    Ex, Ey = Epoints(X,Y, varaukset)
    ax.streamplot(X,Y,Ex,Ey)
    ax.set_aspect('equal')
    plt.title("Many pointlines using streamplot")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")    

# testing the line integraalihässäkkä
def simpslinetesti(intpoints = 10):
   
    x = np.linspace(-4,4,50)
    y = np.linspace(-4,4,50)
    X,Y = np.meshgrid(x,y)
    Ex, Ey = np.zeros_like(X), np.zeros_like(Y) 
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    alku = np.array([-2,0])
    loppu = np.array([2,0])
    ax.plot([-2, 2], [0,0], '-r')
    Ex, Ey = Esimpsline(alku, loppu, intpoints,3, X, Y, Ex, Ey)
    
    alku = np.array([0,0])
    loppu = np.array([0,3])
    ax.plot([0, 0], [0,3], '-b')
    Ex, Ey = Esimpsline(alku, loppu, intpoints,-1, X, Y, Ex, Ey)
    
    alku = np.array([0,0])
    loppu = np.array([3,3])
    ax.plot([0, 3], [0,3], '-b')
    Ex, Ey = Esimpsline(alku, loppu, intpoints,-1, X, Y, Ex, Ey)
    
    alku = np.array([-1,-4])
    loppu = np.array([1,-4])
    ax.plot([-1, 1], [-4,-4], '-b')
    Ex, Ey = Esimpsline(alku, loppu, intpoints,-1, X, Y, Ex, Ey)

    ax.streamplot(X,Y,Ex,Ey)
    plt.title("Simpson integrated lines using streamplot and {} points per line".format(intpoints))
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")    
    
    
    

def main():
    face()
    testRod(2,1,1e-6,10)
    linestest()
    simpslinetesti(13)
 
    
    
    
if __name__=="__main__":
    main()
    
