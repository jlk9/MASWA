#include "MASW.h"

// Written by Joseph Kump (josek97@vt.edu). Last modified 06/21/2019

/* Computes the theoretical curve given ground parameters, as well as the misfit with experimentally
   derived velocity data. Returns the misfit between the curve's theoretical and experiemental dispersion curves.
   
   Inputs:
   curve is a dispersion curve struct (curve_t) containing wavelengths, velocities, and
   ground model parameters. Consult the header file for all of its variables.
   
*/
dfloat MASWA_inversion(curve_t *curve){
    
    // Compute the theoretical curve:
    MASWA_theoretical_dispersion_curve(curve);

    // Get our error:
    dfloat e = MASWA_misfit(curve);

    return e;
}
