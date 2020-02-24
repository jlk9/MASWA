#include "MASW.cuh"

/* Sets the experimentally derived variables for the dispersion curve.
    Since we will eventually re-implement MPI for CUDA, we are still
    treating this as an MPI setup but with rank premanantly set to 0
    and size set to 1. */
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
*/
int validNumberCheck(curve_t *curve){
    
    for (int i=0; i<curve->n+1; ++i){
        
        if (curve->alpha[i] > 5000.0){
            printf("Error: alpha[%d] is greater than 5000 m/s\n", i);
            return 0;
        }
        if (curve->beta[i] > 5000.0){
            printf("Error: beta[%d] is greater than 5000 m/s\n", i);
            return 0;
        }
        if (i != curve->n && curve->h[i] < 0.0){
            printf("Error: h[%d] is negative\n", i);
            return 0;
        }
        if (curve->rho[i] < 0.0){
            printf("Error: rho[%d] is negative\n", i);
            return 0;
        }
    }
    for (int i=0; i<curve->velocities_length; ++i){
        if (curve->c_test[i] > 5000.0){
            printf("Error: c_test[%d] is greater than 5000 m/s\n", i);
            return 0;
        }
    }
    for (int i=0; i<curve->curve_length; ++i){
        if (curve->lambda_curve0[i] < 0.0){
            printf("Error: lambda_curve0[%d] is negative\n", i);
            return 0;
        }
    }
    
    return 1;
}
