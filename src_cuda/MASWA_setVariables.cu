#include "MASW.cuh"

// Written by Joseph Kump (josek97@vt.edu). Last modified 11/08/2019

/* Sets the experimentally derived variables for the dispersion curve.
    Since we will eventually re-implement MPI for CUDA, we are still
    treating this as an MPI setup but with rank premanantly set to 0
    and size set to 1.

    Inputs:
    curve           a curve_t object
    c_test          an array of test velocities
    h               thicknesses of the model's layers
    alpha           array of compressional wave velocities
    beta            array of shear wave velocities
    rho             array of layer densities
    n               number of finite thickness layers
    c_length        length of c_test
    l_length        length of lambda_curve0 and c_curve0
    c_curve0        the experimentally derived velocities of the dispersion curve
    lambda_curve0   the wavelengths of the dispersion curve
    
    Output:
    1 if run correctly, 0 if one of the input variables is unrealistic
    Note: unless one of the inout variables is unrealistic, this function sets all of the
        curve_t variables except for lambda_t and curve_t (the theoretical wavelengths and
        velocities of the dispersion curve).
*/
int MASWA_setVariables(curve_t *curve, dfloat *c_test, dfloat *h, dfloat *alpha, dfloat *beta, dfloat *rho, int n,
                       int c_length, int l_length, dfloat *c_curve0, dfloat *lambda_curve0){

    int rank, size;
    // Since MPI and CUDA have not yet been integrated, we are treating
    // rank as 0 and size as 1:
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //MPI_Comm_size(MPI_COMM_WORLD, &size);

    rank = 0;
    size = 1;
    
    if (rank > -1){
        /* Setting variables on this rank: */
        curve->n                    = n;
        curve->velocities_length    = c_length;
        curve->total_length         = l_length;
        curve->curve_length         = l_length / size;

        if (rank < l_length % size){
            ++curve->curve_length;
        }


        curve->c_test   = c_test;
    
        curve->h        = h;
        curve->alpha    = alpha;
        curve->beta     = beta;
        curve->rho      = rho;

        curve->c_curve0         = (dfloat*) calloc(curve->curve_length, sizeof(dfloat));
        curve->lambda_curve0    = (dfloat*) calloc(curve->curve_length, sizeof(dfloat));
        
        // Here we allocate the memory for the theoretical dispersion curve:
        curve->c_t              = (dfloat*) calloc(curve->curve_length, sizeof(dfloat));
        
        // This partitions the dispersion curve modularly:
        for (int i=0; i<curve->curve_length; ++i){
            curve->c_curve0[i] = c_curve0[rank + i*size];
            curve->lambda_curve0[i] = lambda_curve0[rank + i*size];
        }

        int check = validNumberCheck(curve);
    
        if (check == 0){
            printf("Error: data is invalid, empty array returned.\n");
        }
        return check;
    }
    
    return 1;
}

/* Check that all data for the inversion are "valid": velocities should be below 5000 m/s,
and layer thicknesses, densities, and wavelengths should not be negative.

    Inputs:
    curve       the dispersion curve struct
    
    Output:
    1 if all entries are valid, 0 otherwise.
*/
int validNumberCheck(curve_t *curve){
    
    for (int i=0; i<curve->n+1; ++i){
    
        // First we check none of the model velocities are too great:
        if (curve->alpha[i] > 5000.0){
            printf("Error: alpha[%d] is greater than 5000 m/s\n", i);
            return 0;
        }
        if (curve->beta[i] > 5000.0){
            printf("Error: beta[%d] is greater than 5000 m/s\n", i);
            return 0;
        }
        // Then we check for negative thicknesses and densities (obviously bad):
        if (i != curve->n && curve->h[i] < 0.0){
            printf("Error: h[%d] is negative\n", i);
            return 0;
        }
        if (curve->rho[i] < 0.0){
            printf("Error: rho[%d] is negative\n", i);
            return 0;
        }
    }
    // Make sure none of the test velocities are unrealistic:
    for (int i=0; i<curve->velocities_length; ++i){
        if (curve->c_test[i] > 5000.0){
            printf("Error: c_test[%d] is greater than 5000 m/s\n", i);
            return 0;
        }
    }
    // Make sure no wavelengths are negative:
    for (int i=0; i<curve->curve_length; ++i){
        if (curve->lambda_curve0[i] < 0.0){
            printf("Error: lambda_curve0[%d] is negative\n", i);
            return 0;
        }
    }
    
    return 1;
}
