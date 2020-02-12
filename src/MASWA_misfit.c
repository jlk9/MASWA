#include "MASW.h"

/* Computes the relative error between an experimentally derived dispersion curve and
   a theoretical dispersion curve.
*/
dfloat MASWA_misfit(curve_t *curve){

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    dfloat temp = compute_misfit(curve->curve_length, curve->c_t, curve->c_curve0);

    dfloat total_misfit;
    
    // Need to change to float if using single precision
    MPI_Reduce(&temp, &total_misfit, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0){
        curve->misfit = total_misfit * 100.0 / (dfloat) curve->total_length;
    }
    else{
        curve->misfit = temp * 100.0 / (dfloat) curve->total_length;
    }

    return curve->misfit;
}

/* Computes the error values for the entries of the dispersion curve on this rank.
    This is then added to the other ranks in the parent function above.
*/
dfloat compute_misfit(int curve_length, dfloat *c_t, dfloat *c_curve0){

    dfloat error = 0.0;
    for (int i=0; i<curve_length; ++i){
        error = error + sqrt((c_curve0[i]-c_t[i])*(c_curve0[i]-c_t[i])) / c_curve0[i];
    }

    return error;

}
