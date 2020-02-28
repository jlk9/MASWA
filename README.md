# MASWA
Accelerated algorithms for Multichannel Analysis of Surface Waves (MASW) inversion using MPI and CUDA.

## Separate Versions:

Currently the MPI and Cuda implementations are two separate algorithms, so the user chooses which form of parallelization best suites their resources and uses that version of MASWA (MPI if they have many CPU cores, Cuda if they have a GPU). The header and c files for the MPI version are in `include` and `src`, respectively, while the files for the Cuda version are in `include_cuda` and `src_cuda`.

Both versions are compiled using the `compilation.py` script, written in Python. Both versions mostly use C code, with some C++ functionality to handle complex numbers. The Cuda version obviously uses Cuda for GPU kernels.

## Requirements:

Although both versions use the same compilation script, additional requirements for one version are not necessary to run the other. For example, a GPU is not needed at all to compile and run the MPI algorithm, while no version of Open MPI or Open RTE is needed to run the Cuda version.

###### Requirements for both versions:
- a CPU with one or more cores
- python 3 to run the `compilation.py` script (it was tested with Python 3.7.3)
- g++ compiler (tested and proven to work on: GCC version 5.2.0, Ubuntu version 7.4.0, and Apple LLVM version 10.0.1)
- some common C libraries: `stdio.h`, `stdlib.h`, `string.h`, `math.h`, and `time.h`
- the c++ `complex` library for complex numbers

###### Additional requirements for  MPI version:
- Open MPI (versions that have been tested are 1.8.5, 2.1.1, and 3.1.3)
- alternatively, Open RTE also works for running the MPI code (tested versions are 2.1.1 and 3.1.3)
- ideally, as many CPU cores as processes you wish to use for running the code

###### Additional requirements for Cuda version:
- an Nvidia GPU (the code was tested on an Nvidia Quadro K620, and is designed to work with older GPUs as well as newer ones)
- the nvcc Cuda compiler driver (tested on V9.1.85)
- the `cuComplex` library in Cuda

## Running MASWA

###### MPI version:
To compile MPI MASWA, first make sure all of the required software is downloaded. Then enter this command:

`python compilation.py mpi`

to compile the mpi code. From there you can run it with this command is using Open MPI:

`mpirun -np <number of processes> ./maswa_mpi`
  
If using Open RTE instead, then replace `mpirun` with `mpiexec`. To delete the `.o` files generated by compilation, run:

`python compilation.py mpi_clean`

Currently the particular dispersion curve data and model you wish to run must be set up in the `main` function. A future enhancement will be to add file IO and command line arguments to the executable for usability.

###### Cuda version:
To compile Cuda MASWA, first make sure all of the required software is downloaded. Then enter this command:

`python compilation.py cuda`

to compile the cuda code. From there you can run it with this command:

`./maswa_cuda`
  
To delete the `.o` files generated by compilation, run:

`python compilation.py cuda_clean`

Like the MPI version, the Cuda version requires the desired tests and data to be placed in the `main` function. It is possible the Cuda tags in `compilation.py` may not work for your version of Cuda, so they may need to be changed.

## Citation:
This code was based off of MASWaves, an implementation of MASW written in Matlab.

Ólafsdóttir, Elín Ásta & Erlingsson, Sigurdur & Bessason, Bjarni. (2018). OPEN SOFTWARE FOR ANALYSIS OF MASW DATA. 
