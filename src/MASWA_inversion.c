#include "MASW.h"

/* Computes the theoretical curve given ground parameters, as well as the misfit with experimentally
   derived velocity data. Returns the misfit between the curve's theoretical and experiemental dispsersion curves.
*/
dfloat MASWA_inversion(curve_t *curve){
    
    // Compute the theoretical curve:
    MASWA_theoretical_dispersion_curve(curve);

    // Get our error:
    dfloat e = MASWA_misfit(curve);

    return e;
}
