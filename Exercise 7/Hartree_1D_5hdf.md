#! /usr/bin/env python3

"""
Hartree code for N-electron 1D harmonic quantum dot

- Related to FYS-4096 Computational Physics 
- Test case using simpson integrations should give:

    Total energy      11.502873299221452
    Kinetic energy    3.622113606711247
    Potential energy  7.880759692510205

    Density integral  3.9999999999999996

- Job description in short (more details in the pdf-file): 
  -- Problem 1: add/fill needed functions and details
  -- Problem 2: Include input and output as text file
  -- Problem 3: Include input and output as HDF5 file 
"""
from numpy import *
from matplotlib.pyplot import *
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps
import h5py
import os

FILENAMESAVE = "perjantai.hdf5"
FILENAMELOAD = "perjantai.hdf5"
SAVING = True
LOADING = True

TULOSTETAANKEHITYS = True

def hartree_potential(ns,x):
    """ 
    Hartree potential using Simpson integration 
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
    global ee_coef
    
    """ 1D electron-electron interaction """
    return ee_coef[0] / sqrt(x**2 + ee_coef[1])


def ext_potential(x,m=1.0,omega=1.0):
    """ 1D harmonic quantum dot """
    return 0.5*m*omega**2*x**2


def density(psis):
    ns=zeros((len(psis[0]),))

    for i in range(len(ns)):
        
        for idx in range(len(psis[:])):
            
            ns[i] += psis[idx][i] * conj(psis[idx][i])
             
    return ns
    

def initialize_density(x,dx,normalization=1):
    rho=exp(-x**2)
    A=simps(rho,x)
    return normalization/A*rho


def check_convergence(Vold,Vnew,threshold):
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
        
    idx = 1
    for orb in orbitals:
        
        prob = orb * orb.conj()
        ax.plot(x, prob, label = "Density of electron {}".format(idx))
        idx += 1
        
        
def save_data_to_hdf5_file(fname, orbitals, density, N_e, occ, grid, ee_coefs):

    f = h5py.File(fname,"w")
    
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
    V_SIC = []
    for i in range(len(orbitals)):
        V_SIC.append(-hartree_potential(abs(orbitals[i])**2,x))
    return V_SIC
   
         
def normalize_orbital(evec,x):
    """ Normalize orbitals to one """
    
    integrand = evec * evec.conj()
    integral = simps(integrand, x)
    
    return evec / sqrt(integral)
 
    
def kinetic_hamiltonian(x):
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
    ee_coef = [1.0, 1.0]
    # number of electrons
    N_e = 6
    # 1D occupations each orbital can have max 2, i.e., spin up and spin down
    # e.g., occ = [0,0,1,1] means lowest up and down, next lowest up and down
    #       occ = [0,1,2,3] means all spin up
    
    # S = 3
    occ = [0,1,2,3,4,5]
    # S = 0
 
    
    # occ = [0,0,1,1,2,2]
    
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
    dx = x[1]-x[0]
    T = kinetic_hamiltonian(x)
    Vext = ext_potential(x)

    filename = FILENAMELOAD
              
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
        ns=initialize_density(x,dx,N_e)

    print('Density integral        ', simps(ns,x))
    print(' -- should be close to  ', N_e)
    
    print('\nCalculating initial state')
    Vhartree=hartree_potential(ns,x)
    VSIC=[]
    
    if LOADING:                          
        VSIC=calculate_SIC(orbitals,x)
        
    else:
        for i in range(N_e):
            VSIC.append(ns*0.0)            
        
    Veff=sp.diags(Vext+Vhartree,0)
    H=T+Veff
    
    if TULOSTETAANKEHITYS:
        fig = figure(figsize=(13,13))
        ax = fig.gca()
    
    iter = 0
    for i in range(maxiters):
        print('\n\nIteration #{0}'.format(i))
        orbitals=[]
        for i in range(N_e):
            print('  Calculating orbitals for electron ', i+1)
            eigs, evecs = sla.eigs(H+sp.diags(VSIC[i],0), k=N_e, which='SR')
            eigs=real(eigs)
            evecs=real(evecs)
            print('    eigenvalues', eigs)
            evecs[:,occ[i]]=normalize_orbital(evecs[:,occ[i]],x)
            orbitals.append(evecs[:,occ[i]])
        Veff_old = 1.0*Veff
        ns=density(orbitals)
        Vhartree=hartree_potential(ns,x)
        VSIC=calculate_SIC(orbitals,x)
        Veff_new=sp.diags(Vext+Vhartree,0)
        
        if TULOSTETAANKEHITYS:
            ax.plot(x, ns, label = "{}".format(iter))
            iter += 1
        
        if check_convergence(Veff_old,Veff_new,threshold):
            break
        else:
            """ Mixing the potential """
            Veff= mix * Veff_old +  (1 - mix) * Veff_new
            H = T+Veff

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

    if SAVING:
        filename = FILENAMESAVE
        save_data_to_hdf5_file(filename, orbitals, ns, N_e, occ, x, ee_coef)
            
    fig = figure(figsize = (13,13))
    ax = fig.gca()
    ax.plot(x,abs(ns), label = "Total electron density")
    ax.set_xlabel(r'$x$ (a.u.)')
    ax.set_ylabel(r'$n(x)$ (1/a.u.)')

    ax.set_title("N-electron density for N={} and tolerance {} and occ {} and ee_coef {}".format(N_e, threshold, occ, ee_coef))
    
    energytext = "Total energy {:2.3f} Ha \n Kinetic energy {:2.3f} Ha \n Potential energy {:2.3f} Ha".format(E_tot, E_kin, E_pot)            
    ax.text(0.2,0.9,energytext, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    
    plot_orbitals(orbitals, x, ax)
    legend()
    show()


if __name__=="__main__":
    main()
