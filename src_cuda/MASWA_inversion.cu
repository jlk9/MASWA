#include "MASW.cuh"

// Written by Joseph Kump (josek97@vt.edu). Last modified 01/27/2020

/* This function runs the entire inversion process, utilizing the GPU to compute determinants,
   find the first sign change, and calculate the misfit between the theoretical and experimentally
   derived curves.
   
   We assume the model parameters have already been set in MASWA_setVariables.
   
    Inputs:
    curve the dispersion curve struct, described in MASW.h
    
    Outputs:
    0 if run correctly, but the theoretical wavelengths, velocities, and misfit are all
        set in the curve_t parameters
*/
dfloat MASWA_inversion_CUDA(curve_t *curve){

    MASWA_theoretical_dispersion_curve_CUDA(curve);

    return 0.0;

}
