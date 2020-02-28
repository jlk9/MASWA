#include "MASW.cuh"

// Written by Joseph Kump (josek97@vt.edu). Last modified 01/27/2020

/* This function runs the entire inversion process, utilizing the GPU to compute determinants,
   find the first sign change, and calculate the misfit between the theoretical and experimentally
   derived curves.
   
    Inputs:
    curve the dispersion curve struct, described in MASW.h
*/

/* We assume the curve's variables have already been set using set_variables.
*/
dfloat MASWA_inversion_CUDA(curve_t *curve){

    MASWA_theoretical_dispersion_curve_CUDA(curve);

    return 0.0;

}
