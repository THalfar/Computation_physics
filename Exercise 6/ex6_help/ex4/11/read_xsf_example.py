from numpy import *
from spline_class import spline
from matplotlib.pyplot import *
from scipy.integrate import simps

def read_example_xsf_density(filename):
    lattice=[]
    density=[]
    grid=[]
    shift=[]
    i=0
    start_reading = False
    with open(filename, 'r') as f:
        for line in f:
            if "END_DATAGRID_3D" in line:
                start_reading = False
            if start_reading and i==1:
                grid=array(line.split(),dtype=int)
            if start_reading and i==2:
                shift.append(array(line.split(),dtype=float))
            if start_reading and i==3:
                lattice.append(array(line.split(),dtype=float))
            if start_reading and i==4:
                lattice.append(array(line.split(),dtype=float))
            if start_reading and i==5:
                lattice.append(array(line.split(),dtype=float))
            if start_reading and i>5:            
                density.extend(array(line.split(),dtype=float))
            if start_reading and i>0:
                i=i+1
            if "DATAGRID_3D_UNKNOWN" in line:
                start_reading = True
                i=1
    
    rho=zeros((grid[0],grid[1],grid[2]))
    ii=0
    for k in range(grid[2]):
        for j in range(grid[1]):        
            for i in range(grid[0]):
                rho[i,j,k]=density[ii]
                ii+=1

    # convert density to 1/Angstrom**3 from 1/Bohr**3
    a0=0.52917721067
    a03=a0*a0*a0
    rho/=a03
    return rho, array(lattice), grid, shift


def main():
    filename = 'dft_chargedensity2.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    
    #print(lattice)
    
    dlattice=1.0*lattice
    for i in range(3):
        dlattice[i,:]/=(grid[i]-1)
    dV = abs(linalg.det(dlattice))
    print('Num. electrons: ', sum(sum(sum(rho[0:-1,0:-1,0:-1])))*dV)

    print('Reciprocal lattice as [b1,b2,b3]: ')
    print(transpose(2.0*pi*linalg.inv(transpose(lattice))))

    if filename=='dft_chargedensity1.xsf':
        x = linspace(0,lattice[0,0],grid[0])
        y = linspace(0,lattice[1,1],grid[1])
        z = linspace(0,lattice[2,2],grid[2])

        int_dx = simps(rho,x=x,axis=0)
        int_dxdy = simps(int_dx,x=y,axis=0)
        int_dxdydz = simps(int_dxdy,x=z)
        print('  Simpson int = ',int_dxdydz)

        spl3d=spline(x=x,y=y,z=z,f=rho,dims=3)

        r1=array([0.1,0.1,2.8528])
        r2=array([4.45,4.45,2.8528])
        t = linspace(0,1,500)
        f = zeros(shape=shape(t))
        for i in range(len(t)):
            xx=r1+t[i]*(r2-r1)
            f[i]=spl3d.eval3d(xx[0],xx[1],xx[2])

        plot(t*sqrt(sum((r2-r1)**2)),f)
        show()
    else:
        inv_cell=linalg.inv(transpose(lattice))
        x = linspace(0,1.,grid[0])
        y = linspace(0,1.,grid[1])
        z = linspace(0,1.,grid[2])
        
        int_dx = simps(rho,x=x,axis=0)
        int_dxdy = simps(int_dx,x=y,axis=0)
        int_dxdydz = simps(int_dxdy,x=z)
        print('  Simpson int = ',int_dxdydz*abs(linalg.det(lattice)))

        spl3d=spline(x=x,y=y,z=z,f=rho,dims=3)
        
        r1=array([-1.4466, 1.3073, 3.2115])
        r2=array([1.4361, 3.1883, 1.3542])
        t = linspace(0,1,500)
        f = zeros(shape=shape(t))
        for i in range(len(t)):
            xx=inv_cell.dot(r1+t[i]*(r2-r1))
            f[i]=spl3d.eval3d(xx[0],xx[1],xx[2])

        figure()
        plot(t*sqrt(sum((r2-r1)**2)),f)

        r1=array([2.9996, 2.1733, 2.1462])
        r2=array([8.7516, 2.1733, 2.1462])
        t = linspace(0,1,500)
        f = zeros(shape=shape(t))
        uvec=ones(r1.shape)
        for i in range(len(t)):
            xx=mod(inv_cell.dot(r1+t[i]*(r2-r1)),uvec)
            f[i]=spl3d.eval3d(xx[0],xx[1],xx[2])

        figure()
        plot(t*sqrt(sum((r2-r1)**2)),f)
        show()
    

if __name__=="__main__":
    main()



