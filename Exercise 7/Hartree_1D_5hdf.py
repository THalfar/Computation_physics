#! /usr/bin/env python3

"""
Hartree code for N-electron 1D harmonic quantum dot

- Related to FYS-4096 Computational Physics 
- Test case using simpson integrations should give:

    Total energy      11.502873299221452
    Kinetic energy    3.622113606711247
    Potential energy  7.880759692510205

    Density integral  3.9999999999999996


"""
from numpy import *
from matplotlib.pyplot import *
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps
import h5py
import os

FILENAMESAVE = "sunnuntai2.hdf5" # Save filename
FILENAMELOAD = "sunnuntai2.hdf5" # Load filename
SAVING = True # If saving
LOADING = False # If loading

TULOSTETAANKEHITYS = False # For visualization of SCF convergence

def hartree_potential(ns,x):
    """
    Hartree potential calculation

    Parameters
    ----------
    ns : np.array
        Electron density vector
    x : np.array
        Grid 

    Returns
    -------
    Vhartree : np.array
        Hartree potential 

    """
    
    Vhartree=0.0*ns
    for ix in range(len(x)):
        r = x[ix]
        f = 0.0*x
        for ix2 in range(len(x)):
            rp = x[ix2]
            f[ix2]=ns[ix2]*ee_potential(r-rp)
        Vhartree[ix]= simps(f, x)
    return Vhartree


def ee_potential(x):
    """
    Electron potential calculation

    Parameters
    ----------
    x : np.array
        Grid where operate

    Returns
    -------
    float
        electron-electron potential 

    """
    global ee_coef  # global variable, describing e-e potential 
        
    return ee_coef[0] / sqrt(x**2 + ee_coef[1])


def ext_potential(x,m=1.0,omega=1.0):
    """
    External potential of quantum point

    Parameters
    ----------
    x : np.array
        Grid where operate
    m : float, optional
        Magnitude of potential. The default is 1.0.
    omega : float, optional
        Omega parameter of potential. The default is 1.0.

    Returns
    -------
    TYPE
        Quantum point potential

    """
    return 0.5*m*omega**2*x**2


def density(psis):
    """
    Calculates the electron density of different orbitals

    Parameters
    ----------
    psis : np.array
        Orbitals of electrons

    Returns
    -------
    ns : np.array
        Electron density

    """
    ns=zeros((len(psis[0]),))

    for i in range(len(ns)):
        
        for idx in range(len(psis[:])):
            
            ns[i] += psis[idx][i] * conj(psis[idx][i])
             
    return ns
    

def initialize_density(x,dx,normalization=1):
    """
    Initial electron density 

    Parameters
    ----------
    x : np.array
        Grid where we life
    dx : float
        Grid spaceing
    normalization : int, optional
        Normalization constant. The default is 1.

    Returns
    -------
    np.array
        Electron density at starting

    """
    # Gaussian starting point
    rho=exp(-x**2)
    A=simps(rho,x)
    return normalization/A*rho


def check_convergence(Vold,Vnew,threshold):
    """
    Convergence check i.e when to stop

    Parameters
    ----------
    Vold : np.array
        Old potential
    Vnew : np.array
        New potential
    threshold : float
        Threshold of stopping

    Returns
    -------
    converged : bool
        If convergence is reached

    """
    difference_ = amax(abs(Vold-Vnew))
    print('  Convergence check:', difference_)
    converged=False
    if difference_ <threshold:
        converged=True
    return converged


def diagonal_energy(T,orbitals,x):
    """ 
    Calculate diagonal energy
    (using Simpson)
    """
    Tt=sp.csr_matrix(T)
    E_diag=0.0
    
    for i in range(len(orbitals)):
        evec=orbitals[i]
        E_diag+=simps(evec.conj()*Tt.dot(evec),x)
    return E_diag


def offdiag_potential_energy(orbitals,x):
    """ 
    Calculate off-diagonal energy
    (using Simpson)
    """
    U = 0.0
    for i in range(len(orbitals)-1):
        for j in range(i+1,len(orbitals)):
            fi = 0.0*x
            for i1 in range(len(x)):
                fj = 0.0*x
                for j1 in range(len(x)):
                    fj[j1]=abs(orbitals[i][i1])**2*abs(orbitals[j][j1])**2*ee_potential(x[i1]-x[j1])
                fi[i1]=simps(fj,x)
            U+=simps(fi,x)
    return U


def plot_orbitals(orbitals, x, ax):
    """
    Plots orbitals of different electrons

    Parameters
    ----------
    orbitals : np.array
        Orbitals of electrons
    x : np.array
        Grid where we life
    ax : matplotlib axis
        Axis where to plot

    Returns
    -------
    None.

    """
    idx = 1
    for orb in orbitals:
        
        prob = orb * orb.conj()
        ax.plot(x, prob, label = "Density of electron {}".format(idx))
        idx += 1


def save_data_to_hdf5_file(fname, orbitals, density, N_e, occ, grid, ee_coefs):
    """
    Save all data to hdf5 files. Much nicer than ASCII!

    Parameters
    ----------
    fname : string
        Filename where to save
    orbitals : np.array
        Electrons orbitals
    density : np.array
        Electron density
    N_e : int
        Number of electrons
    occ : list
        Occupied states of electrons i.e spins here
    grid : np.array
        Grid where we life
    ee_coefs : list
        electron-electron potential calculations constants
        
    Returns
    -------
    None.

    """

    f = h5py.File(fname,"w")
    print("Saving to file: {}".format(fname))
    
    gset = f.create_dataset("grid",data=grid,dtype='f')
    gset.attrs["info"] = '1D grid'
    
    oset = f.create_dataset("orbitals",shape=(len(grid),N_e),dtype='f')
    oset.attrs["info"] = '1D orbitals as (len(grid),N_electrons)'
    for i in range(len(orbitals)):
        oset[:,i]=orbitals[i]
        
    occ_set = f.create_dataset("occ", data=occ, dtype=int)
    occ_set.attrs["info"] = "occ"
    
    ee_coefs_set = f.create_dataset("ee_coefs", data = ee_coefs, dtype = 'f')
    ee_coefs_set.attrs["info"] = "ee_coefs"
    
    density_set = f.create_dataset("density",  data = density, dtype = 'f')
    density_set.attrs["info"] = "density"
    
    f.close()


def calculate_SIC(orbitals,x):
    """
    Calculates self-interaction-correction SIC i.e electron cannot 
    lift itself by hair

    Parameters
    ----------
    orbitals : np.array
        Orbitals of electrons
    x : np.array
        Grid where we life

    Returns
    -------
    V_SIC : list
        Self interaction potential of different orbitals

    """
    V_SIC = []
    for i in range(len(orbitals)):
        V_SIC.append(-hartree_potential(abs(orbitals[i])**2,x))
    return V_SIC
   
         
def normalize_orbital(evec,x):
    """
    Normalizes orbitals to one i.e electron must be somewhere

    Parameters
    ----------
    evec : np.array
        Electron psi 
    x : np.array
        Grid

    Returns
    -------
    np.array
        Normalized electron orbitals

    """
    
    integrand = evec * evec.conj()
    integral = simps(integrand, x)
    
    return evec / sqrt(integral)
 
    
def kinetic_hamiltonian(x):
    """
    Kinetic hamiltonian function

    Parameters
    ----------
    x : np.array
        Grid 

    Returns
    -------
    H0 : np.array
        Hamiltonian reprending 1/2m  * nabla**2  (?)
        

    """
    grid_size = x.shape[0]
    dx = x[1] - x[0]
    dx2 = dx**2
    
    H0 = sp.diags(
        [
            -0.5 / dx2 * np.ones(grid_size - 1),
            1.0 / dx2 * np.ones(grid_size),
            -0.5 / dx2 * np.ones(grid_size - 1)
        ],
        [-1, 0, 1])
    return H0


def main():

     # --- Setting up the system etc. ---
    global ee_coef
    
    # e-e potential parameters [strenght, smoothness]
    ee_coef = [0.0, 1.0]
 
    # number of electrons
    N_e = 6
 
    # S = 3
    occ = [0,1,2,3,4,5]
    # S = 0
    # occ = [0,0,1,1,2,2]
    
    # For problems 1-3
    # occ = [0,1,2,3]
    # occ = [0,0,1,1]
    
    # grid
    x=linspace(-4,4,120)
    
    # threshold
    threshold=1.0e-6
    
    # mixing value
    mix=0.2
    
    # maximum number of iterations
    maxiters = 100
    
    # --- End setting up the system etc. ---

    # Initialization and calculations start here
    dx = x[1]-x[0] # grid difference
    T = kinetic_hamiltonian(x) # kinetic hamiltonian of this grid
    Vext = ext_potential(x) # used external potential

    filename = FILENAMELOAD # from which file load
              
    # Load if loading 
    if LOADING:

        print("Loading from file {}".format(filename))
        print("*** *** ***")
        
        f = h5py.File(filename, 'r')
        print('Keys in hdf5 file: ',list(f.keys()))
        
        x = array(f["grid"])
        orbs = array(f["orbitals"])
        orbitals = []
        
        for i in range(len(orbs[0,:])):
            orbitals.append(orbs[:,i])
            
        occ = list(f["occ"])
        ee_coef = list(f["ee_coefs"])
        ns = array(f["density"])
        
        f.close()
  
    else:
        # If not load use these
        ns=initialize_density(x,dx,N_e)

    print('Density integral        ', simps(ns,x))
    print(' -- should be close to  ', N_e)
    
    print('\nCalculating initial state')
    Vhartree=hartree_potential(ns,x)
    VSIC=[]
    
    # Load or set to zero
    if LOADING:                          
        VSIC=calculate_SIC(orbitals,x)
        
    else:
        for i in range(N_e):
            VSIC.append(ns*0.0)            
        
    Veff=sp.diags(Vext+Vhartree,0)  # Starting potential part 1) in SCF
    H=T+Veff
    
    # If want to visualize how SCF converges 
    if TULOSTETAANKEHITYS:
        fig = figure(figsize=(13,13))
        ax = fig.gca()
    
    ite = 0 # used for plotting convergence
    # Iterate until convergence or maxiter is fulfilled
    for i in range(maxiters):
        print('\n\nIteration #{0}'.format(i))
        orbitals=[]
        for i in range(N_e):
            print('  Calculating orbitals for electron ', i+1)
            # Solve eigenvalues and vectors of different orbitals part 2) in SCF
            eigs, evecs = sla.eigs(H+sp.diags(VSIC[i],0), k=N_e, which='SR') # Include SIC terms in energys
            eigs=real(eigs)
            evecs=real(evecs)
            print('    eigenvalues', eigs)
            evecs[:,occ[i]]=normalize_orbital(evecs[:,occ[i]],x) # normalize these
            orbitals.append(evecs[:,occ[i]]) # Save to orbitals using spins i.e mixed if two electron at same orbital
        Veff_old = 1.0*Veff # part 3) in SCF
        ns=density(orbitals) # Part 4) in SCF
        Vhartree=hartree_potential(ns,x) # Part 5) in SCF
        VSIC=calculate_SIC(orbitals,x)  # Calculate the SIC of electrons  
        Veff_new=sp.diags(Vext+Vhartree,0) # New potential part 6) in SCF
        
        # If wanting to plot convergence
        if TULOSTETAANKEHITYS:
            ax.plot(x, ns, label = "{}".format(ite))
            ite += 1
            
        # check convergence part 7) SCF
        if check_convergence(Veff_old,Veff_new,threshold):
            break
        else:
            """ Mixing the potential """
            Veff= mix * Veff_old +  (1 - mix) * Veff_new
            H = T+Veff

    # If want to visualize convergence
    if TULOSTETAANKEHITYS:
        ax.set_title("Kehitystä")
        legend()
        show()

    print('\n\n')
    off = offdiag_potential_energy(orbitals,x)
    E_kin = diagonal_energy(T,orbitals,x)
    E_pot = diagonal_energy(sp.diags(Vext,0),orbitals,x) + off
    E_tot = E_kin + E_pot
    print('Total energy     ', E_tot)
    print('Kinetic energy   ', E_kin)
    print('Potential energy ', E_pot) 
    print('\nDensity integral ', simps(ns,x))

    # If saving to file 
    if SAVING:
        filename = FILENAMESAVE
        save_data_to_hdf5_file(filename, orbitals, ns, N_e, occ, x, ee_coef)
            
    fig = figure(figsize = (13,13))
    ax = fig.gca()
    ax.plot(x,abs(ns), label = "Total electron density")
    ax.set_xlabel(r'$x$ (a.u.)')
    ax.set_ylabel(r'$n(x)$ (1/a.u.)')
    # Display energetics
    ax.set_title("N-electron density for N={} and tolerance {} and occ {} and ee_coef {}".format(N_e, threshold, occ, ee_coef))
    
    energytext = "Total energy {:2.3f} Ha \n Kinetic energy {:2.3f} Ha \n Potential energy {:2.3f} Ha".format(E_tot, E_kin, E_pot)            
    ax.text(0.2,0.9,energytext, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    
    plot_orbitals(orbitals, x, ax)    
    legend()
    show()


if __name__=="__main__":
    main()
