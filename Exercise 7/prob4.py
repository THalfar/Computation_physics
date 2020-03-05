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

FILENAMESAVE = "testa"
FILENAMELOAD = "tes"

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

def save_ns_in_ascii(ns,filename):
    s=shape(ns)
    f=open(filename+'_density.txt','w')
    for ix in range(s[0]):
        f.write('{0:12.8f}\n'.format(ns[ix]))
    f.close()
    f=open(filename+'_shape.txt','w')
    f.write('{0:5}'.format(s[0]))
    f.close()
    
    
def save_orbitals_in_ascii(orbitals, filename):
    
    f = open(filename+'_orbitals.txt','w')
    
    for orb in orbitals:
        
        for idx in range(len(orb)):
            
            f.write("{0:12.8f}\n".format(orb[idx]))
            
    f.close


def save_grid_in_ascii(x, filename):
    
    f = open(filename+'_grid.txt','w')
    
    for idx in range(len(x)):        
        f.write("{0:12.8f}\n".format(x[idx]))
        
    f.close

    
def load_orbitals_in_ascii(filename):    
        
    f = open(filename+"_shape.txt", 'r')

    for line in f:
        s = array(line.split(), dtype = int)
    f.close()
    
    d = loadtxt(filename+"_orbitals.txt")
    
    En = int(len(d) / s[0])
    
    orbitals = zeros((En, s[0]))
    
    for e in range(En):
        
        for idx in range(s[0]):
            
            orbitals[e, idx] = d[(e+1) * idx]
                
    return orbitals
    

def load_grid_from_ascii(filename):

    f = open(filename+"_shape.txt", 'r')

    for line in f:
        s = array(line.split(), dtype = int)
    f.close()
    
    x = zeros((s[0],))
    d = loadtxt(filename+"_grid.txt")
    
    for idx in range(s[0]):
        x[idx] = d[idx]
    
    return x
    
def load_ns_from_ascii(filename):
    f=open(filename+'_shape.txt','r')
    for line in f:
        s=array(line.split(),dtype=int)
    f.close()
    ns=zeros((s[0],))
    d=loadtxt(filename+'_density.txt')
    k=0
    for ix in range(s[0]):
        ns[ix]=d[k]
        k+=1
    return ns

def save_data_to_hdf5_file(fname,orbitals,density,N_e,occ,grid,ee_coefs):
    return

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
    N_e = 4

    # 1D occupations each orbital can have max 2, i.e., spin up and spin down
    # e.g., occ = [0,0,1,1] means lowest up and down, next lowest up and down
    #       occ = [0,1,2,3] means all spin up
    occ = [0,1,2,3]

    # grid
    x=linspace(-4,4,120)

    # threshold
    threshold=1.0e-2

    # mixing value
    mix=0.2

    # maximum number of iterations
    maxiters = 100
    # --- End setting up the system etc. ---



    # Initialization and calculations start here
    dx = x[1]-x[0]
    T = kinetic_hamiltonian(x)
    Vext = ext_potential(x)

    """
    #FILL#
    In problems 2 and 3: READ in density / orbitals / etc.
    """
    
    #TODOKYSY onko Vext tallennettava!
    
    filename = FILENAMELOAD
          
    if os.path.isfile(filename + "_shape.txt"):
        
        print("Loading from files {}_*".format(filename))
        print("")
        
        ns=load_ns_from_ascii(filename)
        x = load_grid_from_ascii(filename)
        orbitals = load_orbitals_in_ascii(filename)
        
        

    else:
        ns=initialize_density(x,dx,N_e)

    print('Density integral        ', simps(ns,x))
    print(' -- should be close to  ', N_e)
    
    print('\nCalculating initial state')
    Vhartree=hartree_potential(ns,x)
    VSIC=[]
    for i in range(N_e):
        # Now VSIC is initialized as zero, since there are no orbitals yet
        VSIC.append(ns*0.0)
 
        """
          #FILL#
          In problems 2 and 3 this needs to be modified, since 
          then you have orbitals already at this point !!!!!!!!!!!!
        """

    Veff=sp.diags(Vext+Vhartree,0)
    H=T+Veff
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
        if check_convergence(Veff_old,Veff_new,threshold):
            break
        else:
            """ Mixing the potential """
            Veff= mix * Veff_old +  (1 - mix) * Veff_new
            H = T+Veff

    print('\n\n')
    off = offdiag_potential_energy(orbitals,x)
    E_kin = diagonal_energy(T,orbitals,x)
    E_pot = diagonal_energy(sp.diags(Vext,0),orbitals,x) + off
    E_tot = E_kin + E_pot
    print('Total energy     ', E_tot)
    print('Kinetic energy   ', E_kin)
    print('Potential energy ', E_pot) 
    print('\nDensity integral ', simps(ns,x))

    filename = FILENAMESAVE

    print("Saving to files: {}_*".format(filename))
    
    save_ns_in_ascii(ns,filename)
    save_orbitals_in_ascii(orbitals, filename)
    save_grid_in_ascii(x, filename)
    
    """
    #FILL#
    In problems 2 and 3:
    WRITE OUT to files: density / orbitals / energetics / etc.
    save_ns_in_ascii(ns,'density') etc.
    """

    plot(x,abs(ns))
    xlabel(r'$x$ (a.u.)')
    ylabel(r'$n(x)$ (1/a.u.)')
    title('N-electron density for N={0}'.format(N_e))
    show()

if __name__=="__main__":
    main()
