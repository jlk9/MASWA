#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex>
#include <time.h>

#include "mpi.h"

/* Written by Joseph Kump as a C implementation of MASWaves with MPI parallelization. */

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
dfloat MASWA_inversion(curve_t *curve);

/* MASWA_Ke_halfspace.c */
compfloat *MASWA_Ke_halfspace(dfloat r_alpha, dfloat r_beta, dfloat r_rho, dfloat r_c_test, dfloat r_k);

/* MASWA_Ke_layer.c */
compfloat *MASWA_Ke_layer(dfloat r_h, dfloat r_alpha, dfloat r_beta, dfloat r_rho, dfloat r_c_test, dfloat r_k);

/* MASWA_main.c */
int main(int argc, char **argv);

/* MASWA_misfit.c */
dfloat MASWA_misfit(curve_t *curve);
dfloat compute_misfit(int curve_length, dfloat *c_t, dfloat *c_curve0);

/* MASWA_setVariables.c */
int MASWA_setVariables(curve_t *curve, dfloat *c_test, dfloat *h, dfloat *alpha, dfloat *beta, dfloat *rho, int n,
                       int c_length, int l_length, dfloat *c_curve0, dfloat *lambda_curve0);
int validNumberCheck(curve_t *curve);

/* MASWA_stiffness_matrix.c */
dfloat MASWA_stiffness_matrix(dfloat c_test, dfloat k, dfloat *h, dfloat *alpha, dfloat *beta, dfloat *rho, int n);
compfloat hepta_determinant(compfloat **matrix, int size);
int tooClose(dfloat c_test, dfloat *alpha, dfloat *beta, int length, dfloat epsilon);

/* MASWA_test_cases.c */
int testKeLayer();
int testStiffnessMatrix();
int testProcess();
int testProcess_full();
int testScaling(int curve_size, dfloat wavelength);
int testScaling_full();

/* MASWA_theoretical_dispersion_curve.c */
void MASWA_theoretical_dispersion_curve(curve_t *curve);

/* End of Functions */

