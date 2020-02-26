#include "MASW.h"

// Written by Joseph Kump (josek97@vt.edu). Last modified 09/23/2019.

/* Computes the relative error between an experimentally derived dispersion curve and
   a theoretical dispersion curve.
   
   Inputs:
   curve a dispersion curve struct (see the header file)
   
*/
dfloat MASWA_misfit(curve_t *curve){

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get the error for all dispersion curve entries on this process:
    dfloat temp = compute_misfit(curve->curve_length, curve->c_t, curve->c_curve0);

    dfloat total_misfit;
    
    // Add the summed errors from other processes onto rank 0:
    MPI_Reduce(&temp, &total_misfit, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0){
        // Rank 0 gets the total misfit, other ranks get misfit for their entries only:
        curve->misfit = total_misfit * 100.0 / (dfloat) curve->total_length;
    }
    else{
        curve->misfit = temp * 100.0 / (dfloat) curve->total_length;
    }

    return curve->misfit;
}

/* Computes the error values for the entries of the dispersion curve on this rank.
    This is then added to the other ranks in the parent function above.
    
    Inputs:
    curve_length the length of the dispersion curve
    c_t the array of theoretically generated velocities for the dispersion curve, based
        on the model
    c_curve0 the array of experimentally derived velocities for the dispersion curve
*/
dfloat compute_misfit(int curve_length, dfloat *c_t, dfloat *c_curve0){

    dfloat error = 0.0;
    for (int i=0; i<curve_length; ++i){
        // Add up errors elementwise along the two curves:
        error = error + sqrt((c_curve0[i]-c_t[i])*(c_curve0[i]-c_t[i])) / c_curve0[i];
    }

    return error;

}
