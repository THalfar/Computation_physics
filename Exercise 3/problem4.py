import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

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
    x = np.linspace(-4,4,50)
    y = np.linspace(-4,4,50)
    X,Y = np.meshgrid(x,y)
    
    fig, ax = plt.subplots()
    
    alku = np.array([0,0])
    loppu = np.array([2,-1])
    varaukset = ympyra(alku,np.pi,2*np.pi, 1.5, 500,-1)    
    ax.plot(varaukset[:,0], varaukset[:,1], 'bo')
    
    silmat = np.array([[-0.5,1,0.5],[0.5,1,0.5]])
    varaukset = np.concatenate((varaukset, silmat), axis = 0)
    ax.plot(varaukset[-2:,0], varaukset[-2:,1], 'ro')
    
    Ex, Ey = Epoints(X,Y, varaukset)
    ax.streamplot(X,Y,Ex,Ey)
    ax.set_aspect('equal')
    plt.title("Face using streamplot")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    
    
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
    

def main():
    face()
    testRod(2,1,1e-6,10)
   
    
if __name__=="__main__":
    main()
    
