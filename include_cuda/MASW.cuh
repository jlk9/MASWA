#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex>
#include <time.h>

#include <cuda_runtime.h>
#include "cuComplex.h"
#include "cusparse.h"
#include "cublas.h"
#include "math.h"

#include "MASW.h"

/* Written by Joseph Kump as a C implementation of MASWaves. Will be adapted into
 MPI and Cuda */

/* We're defaulting to using doubles for real and complex numbers, but
 this allows us to switch to floats or long doubles easily:
*/
// Always real:
#define dfloat double
#define dfloatString "double"
#define dfloatFormat "%lf"

/* MASWA_inversion.cu */

/* This function runs the entire inversion process, utilizing the GPU to compute determinants,
   find the first sign change, and calculate the misfit between the theoretical and experimentally
   derived curves. */
dfloat MASWA_inversion_CUDA(curve_t *curve);

/* MASWA_ke.cu */

/* Fills in entries of all the stiffness matrices for each wavelength and test velocity,
    breaking up the task so each thread (across multiple blocks) sets up one stiffness
    matrix. */
__global__ void kernel_generate_stiffness_matrices(dfloat *c_test, dfloat *lambda, dfloat *h, dfloat *alpha, dfloat *beta, dfloat *rho, int n, int velocities_length, int curve_length, cuDoubleComplex **matrices);

/* Modifies our test velocities so they are not too close to alpha or beta, causing
    numerical problems (MASWaves does this too). Parallel over one block, with the test
    velocities being divided over blocks. */
__global__ void kernel_too_close(int velocities_length, int nPlus, dfloat *c_test, dfloat *alpha, dfloat *beta, dfloat epsilon);

/* Constructs a Ke Layer in a single CUDA thread, using CUDA complex numbers. Note Ke is length 6.
    This is because we have accounted for symmetry and other redundant entries in the Ke layer, reducing
    the memory requirements and number of computations. */
__device__ void kernel_Ke_layer_symm(dfloat r_h, dfloat r_alpha, dfloat r_beta, dfloat r_rho, dfloat r_c_test, dfloat r_k, cuDoubleComplex *Ke);

/* Creates the information for the final Ke halfspace, using CUDA complex numbers.
    Note Ke is length 3 since it is symmetric, although the halfspace actually has 4 entries. */
__device__ void kernel_Ke_halfspace_symm(dfloat r_alpha, dfloat r_beta, dfloat r_rho, dfloat r_c_test, dfloat r_k, cuDoubleComplex *Ke);

/* MASWA_main.c */

// Since the main function is used in the serial/MPI implementation as well, it is not named here.

/* MASWA_misfit.cu */

/* Runs the kernel for computing the misfit. Not used for the inversion (since it's more efficient to
    just compute the misfit with the dispersion curves while they're already on the GPU), but still
    useful for testing purposes. */
dfloat run_misfit(int curve_length, dfloat *c_t, dfloat *c_curve0);

/* Global kernel, computes the misfit between the real and theoretical curves. Since the dispersion curve
    is typically small, it is usually more efficient to compute the misfit within a single block. */
__global__ void kernel_misfit_00(const int curve_length, dfloat *c_t, dfloat *c_curve0, dfloat *error);

/* Global kernel, computes the misfit between the real and theoretical curves over multiple blocks. Need
    to use kernel_misfit_block_summation below to sum up these block misfits. Generally the single block
    implementation is more efficient, so that is used by default. */
__global__ void kernel_misfit_01(const int curve_length, dfloat *c_t, dfloat *c_curve0, dfloat *error);

/* Sums all the entries in an array and stores them in the first entry. This is used to add up
    multiple blocks' worth of memory for the multi-block misfit. */
__global__ void kernel_misfit_block_summation(const int array_length, dfloat *array);

/* MASWA_setVariables.cu */

// The function MASWA_setVariables is already listed in MASW.h and is essentially a duplicate of that function,
// so it is not declared here. (Eventually when the CUDa and MPI implementations are connected this file will
// be removed.

/* MASWA_stiffness_matrix.cu */

/* Generates the stiffness matrices and fills in their entries, then uses Gaussian
    elimination to make finding their determinants easy. Mostly just allocates memory on
    the GPU, then gets GPU kernels to do the heavy lifting. */
void MASWA_stiffness_matrix_CUDA(curve_t *curve, cuDoubleComplex **d_matrices);

/* Fills in a simple identity matrix for some test cases. Not used as part of MASW (though
    the parameters are modelled after it). */
__global__ void kernel_matrix_fill_in_serial(int velocities_length, int curve_length, int n, cuDoubleComplex **matrices);

/* Assigns matrix pointers to the contiguous chunk of memory where they are stored. This is necessary for the implementation
    utilizing cuBLAS (there may be a method that does not require a kernel, but other things I tried generated a seg fault).
    Since cuBLAS is no longer used, the code may be rewritten to omit this. (Matrices can just be stored as a 1D array without
    pointers, as opposed to a 2D array). */
__global__ void kernel_assign_matrices_to_data(cuDoubleComplex **matrices, cuDoubleComplex *data, int curve_length, int velocities_length, int n);

/* Performs Gaussian elimination on the stiffness matrices by taking advantage of their
    banded structure, unlike in the cuBLAS function. This version puts the current row
    used for elimination into shared memory, improving the speed of accessing it. */
__global__ void kernel_hepta_determinant_CUDA(int curve_length, int velocities_length, int size, cuDoubleComplex **matrices);

/* MASWA_test_cases.cu */

// Functions that are also designed for test cases in the CPU version, like textProcess and testScaling,
// are listed in MASW.h

/* Tests the misfit functions implemented in CUDA, both single block and multi block,
    for accuracy and performance. */
int testMisfitCuda();

/* Tests the sign change across multiple blocks works properly. The vector c_t should
    return [0, 2, 3, 4, 5] */
int testSignChange();

/* Tests that the determinant matrix is set up properly, and the correct sign changes are found.
    TODO: redo this for new determinant functions */
int testFullSignChange();

/* Tests the stiffness matrices are set up properly. */
int testDeterminantMatrixSetup();

/* MASWA_theoretical_dispersion_curve.cu */

/* Carries out the work of MASW inversion. Gets the stiffness matrix determinants, and
    finds the first sign change in each wavelength's array of determinants. */
void MASWA_theoretical_dispersion_curve_CUDA(curve_t *curve);

/* Identifies the first dispersion curve with a sign change in each block of velocity values */
__global__ void kernel_block_sign_change(int *neighbors, int *signChange, int velocities_length, int size, cuDoubleComplex **matrices);

/* Searches through array to find the first sign change for each wavelength.
    This iterates over multiple blocks from kernel_block_sign_change.
    Currently a serial kernel, but it has a negligible effect on the runtime. May be
    parallelized later. */
__global__ void kernel_first_sign_change(dfloat *c_t, dfloat* c_test, int *signChange, int blocksPerWavelength, int curve_length);

/* End of functions */

