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

dfloat MASWA_inversion_CUDA(curve_t *curve);

/* MASWA_ke.cu */

__global__ void kernel_generate_stiffness_matrices(dfloat *c_test, dfloat *lambda, dfloat *h, dfloat *alpha, dfloat *beta, dfloat *rho, int n, int velocities_length, int curve_length, cuDoubleComplex **matrices);
__global__ void kernel_too_close(int velocities_length, int nPlus, dfloat *c_test, dfloat *alpha, dfloat *beta, dfloat epsilon);
__device__ void kernel_Ke_layer_symm(dfloat r_h, dfloat r_alpha, dfloat r_beta, dfloat r_rho, dfloat r_c_test, dfloat r_k, cuDoubleComplex *Ke);
__device__ void kernel_Ke_halfspace_symm(dfloat r_alpha, dfloat r_beta, dfloat r_rho, dfloat r_c_test, dfloat r_k, cuDoubleComplex *Ke);

/* MASWA_main.c */

// Since the main function is used in the serial/MPI implementation as well, it is not named here.

/* MASWA_misfit.cu */

dfloat run_misfit(int curve_length, dfloat *c_t, dfloat *c_curve0);
__global__ void kernel_misfit_00(const int curve_length, dfloat *c_t, dfloat *c_curve0, dfloat *error);
__global__ void kernel_misfit_01(const int curve_length, dfloat *c_t, dfloat *c_curve0, dfloat *error);
__global__ void kernel_misfit_block_summation(const int array_length, dfloat *array);

/* MASWA_setVariables.cu */

// The function MASWA_setVariables is already listed in MASW.h and is essentially a duplicate of that function,
// so it is not declared here. (Eventually when the CUDa and MPI implementations are connected this file will
// be removed.

/* MASWA_stiffness_matrix.cu */

void MASWA_stiffness_matrix_CUDA(curve_t *curve, cuDoubleComplex **d_matrices);
__global__ void kernel_matrix_fill_in_serial(int velocities_length, int curve_length, int n, cuDoubleComplex **matrices);
__global__ void kernel_assign_matrices_to_data(cuDoubleComplex **matrices, cuDoubleComplex *data, int curve_length, int velocities_length, int n);
__global__ void kernel_hepta_determinant_CUDA(int curve_length, int velocities_length, int size, cuDoubleComplex **matrices);

/* MASWA_test_cases.cu */

// Functions that are also designed for test cases in the CPU version, like textProcess and testScaling,
// are listed in MASW.h

int testMisfitCuda();
int testSignChange();
int testFullSignChange();
int testDeterminantMatrixSetup();

/* MASWA_theoretical_dispersion_curve.cu */

void MASWA_theoretical_dispersion_curve_CUDA(curve_t *curve);
__global__ void kernel_block_sign_change(int *neighbors, int *signChange, int velocities_length, int size, cuDoubleComplex **matrices);
__global__ void kernel_first_sign_change(dfloat *c_t, dfloat* c_test, int *signChange, int blocksPerWavelength, int curve_length);

/* End of functions */

