#!/bin/bash

################################################################
## * This script builds available configurations of QMCPACK   ##
##   on Puhti at CSC                   .                      ##
##                                                            ##
## * Execute this script in trunk/                            ##
##   ./config/build_csc_puhti_complex_only.sh                 ##
##                                                            ##
## Last modified: March, 2020                                 ##
################################################################

module load intel/19.0.4
module load hpcx-mpi/2.4.0
module load intel-mkl/2019.0.4
module load StdEnv
module load hdf5/1.10.4
module load fftw/3.3.8-omp
module load boost/1.68.0
module load cmake/3.16.3

CMAKE_FLAGS="-DCMAKE_C_COMPILER=mpicc \ 
             -DCMAKE_CXX_COMPILER=mpicxx"

# Configure and build cpu complex 
echo ""
echo ""
echo "building complex qmcpack for cpu on Puhti"
mkdir -p build_csc_puhti_complex_only
cd build_csc_puhti_complex_only
cmake -DQMC_COMPLEX=1 -DENABLE_SOA=1 $CMAKE_FLAGS ..
make -j 12 
cd ..
ln -sf ./build_csc_puhti_complex_only/bin/qmcpack ./qmcpack_csc_puhti_complex




