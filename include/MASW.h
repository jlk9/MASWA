#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex>
#include <time.h>

#include "mpi.h"

/* Written by Joseph Kump as a C implementation of MASWaves with MPI parallelization.

    Eventually the CUDA and MPI versions of MASWA will be integrated. For now, they
    are each standalone parallelized implementations of the MASW algorithm.
*/

/* We're defaulting to using doubles for real and complex numbers, but
 this allows us to switch to floats or long doubles more easily:
*/

// Real doubles:
#define dfloat double
#define dfloatString "double"
#define dfloatFormat "%lf"

// Complex doubles:
#define compfloat std::complex<double>

/* Struct used to store theoretical and experimentally derived dispersion curve: */
typedef struct{

    /* Array lengths:*/
    int n;                          // Number of finite-thickness layers in ground model
    int curve_length;               // Length of velocity and wavelength vectors
    int velocities_length;          // Number of test velocities
    int total_length;               // Total length of velocity and wavelength vectors

    dfloat misfit;                 // Error between theoretical and experimental velocity curves

    /* Arrays (length of array):*/
    dfloat *c_test;                // Test velocities (velocities_length)

    dfloat *h;                     // Layer thicknesses (n)
    dfloat *alpha;                 // Compressional wave velocities (n+1)
    dfloat *beta;                  // Shear wave velocities (n+1)
    dfloat *rho;                   // Layer densities (n+1)

    dfloat *c_curve0;              // Experimental Rayleigh velocity vector (curve_length)
    dfloat *lambda_curve0;         // Wavelength vector (curve_length)
    dfloat *c_t;                    // Theoretical velocity vector (curve_length)
    dfloat *lambda_t;              // Theoretical wavelength vector (curve_length)
    
}curve_t;

/* Directory for all functions used in MASWA C/MPI implementation, sorted by file. */

/* MASWA_inversion.c */

/* Computes the theoretical curve given ground parameters, as well as the misfit with experimentally
   derived velocity data. Returns the misfit between the curve's theoretical and experiemental dispersion curves. */
dfloat MASWA_inversion(curve_t *curve);

/* MASWA_Ke_halfspace.c */

/* Computes the element stiffness matrix for the half-space (layer n+1) of the stratified earth model that is
 used in the inversion analysis. This halfspace is a 2x2 matrix stored as a 1D array by rows. */
compfloat *MASWA_Ke_halfspace(dfloat r_alpha, dfloat r_beta, dfloat r_rho, dfloat r_c_test, dfloat r_k);

/* MASWA_Ke_layer.c */

/* Computes the element stiffness matrix of the j-th layer (j = 1,...,n) of the stratified earth
 model that is used in the inversion analysis. The stiffness matrix is a 4x4 stored in a 1D Array. */
compfloat *MASWA_Ke_layer(dfloat r_h, dfloat r_alpha, dfloat r_beta, dfloat r_rho, dfloat r_c_test, dfloat r_k);

/* MASWA_main.c */

/* The main function called by the MASWA_mpi executable. Can be used to run MASWA_inversion,
    or one of the test functions available. */
int main(int argc, char **argv);

/* MASWA_misfit.c */

/* Computes the relative error between an experimentally derived dispersion curve and
   a theoretical dispersion curve. */
dfloat MASWA_misfit(curve_t *curve);

/* Computes the error values for the entries of the dispersion curve on this rank.
    This is then added to the other ranks in the parent function above. */
dfloat compute_misfit(int curve_length, dfloat *c_t, dfloat *c_curve0);

/* MASWA_setVariables.c */

/* Sets all the variables for the dispersion curve struct, except for the theoretically
    derived velocities. Just assigning all the struct's entries. */
int MASWA_setVariables(curve_t *curve, dfloat *c_test, dfloat *h, dfloat *alpha, dfloat *beta, dfloat *rho, int n,
                       int c_length, int l_length, dfloat *c_curve0, dfloat *lambda_curve0);

/* Check that all data for the inversion are "valid": velocities should be below 5000 m/s,
and layer thicknesses, densities, and wavelengths should not be negative. Variables
outside these ranges are not useful for MASW. */
int validNumberCheck(curve_t *curve);

/* MASWA_stiffness_matrix.c */

/*The function MASWaves_stiffness_matrix assembles the system stiffness
  matrix of the stratified earth model that is used in the inversion
  analysis and computes its determinant. */
dfloat MASWA_stiffness_matrix(dfloat c_test, dfloat k, dfloat *h, dfloat *alpha, dfloat *beta, dfloat *rho, int n);

/* Helper, used to get the determinant via Guassian row reduction.
    This one takes advantage of the stiffness matrix being heptadiagonal,
    which brings it from O(N^3) to O(N). */
compfloat hepta_determinant(compfloat **matrix, int size);

/* Helper, used to check if c_test is too close to any entries in alpha or beta
   and thus requires modification. */
int tooClose(dfloat c_test, dfloat *alpha, dfloat *beta, int length, dfloat epsilon);

/* MASWA_test_cases.c */

/* Tests correctness of Ke_layer and Ke_halfspace. Not intended to evaluate
    performance or use with multiple processes. */
int testKeLayer();

/* Tests correctness of the stiffness matrix implementation. Not meant to evaluate
    performance or how function works with multiple processes. */
int testStiffnessMatrix();

/* Tests correctness of inversion on a realistic dataset with decreasing wavelengths
    and velocities on the dispersion curve. */
int testProcess();
/* Runs testProcess 11 times (accounting for potential compilation time in first run),
    and gathers the run times to gauge performance. */
int testProcess_full();

/* Tests correctness of inversion on a uniform dataset, to gauge strong and weak scaling
    capabilities of the algorithm with the load imbalancing of variable data. */
int testScaling(int curve_size, dfloat wavelength);

/*Runs test_scaling multiple times over increasing dispersion curves, used to assess weak scaling.
    Collects and prints runtimes to gauge performance. */
int testScaling_full();

/* MASWA_theoretical_dispersion_curve.c */

/* Computes the fundamental mode theoretical dispersion curve. This is done by using the stiffness matrix method,
    which is also used in MASWaves. */
void MASWA_theoretical_dispersion_curve(curve_t *curve);

/* End of Functions */

