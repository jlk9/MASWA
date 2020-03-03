#include "MASW.h"

// Written by Joseph Kump (josek97@vt.edu). Last modified 12/15/2019

/* Computes the fundamental mode theoretical dispersion curve. This is done by using the stiffness matrix method,
    which is also used in MASWaves.
    
    Inputs:
    curve, a dispersion curve struct (curve_t, see the header file) with all of its
        variables set except for the theoretically determined dispersion curve velocities
        and wavelengths. The memory for theoretical velocities and wavelengths is assumed to not 
        be allocated yet. These theoretical velocities and wavelengths will be added by this function.
*/
void MASWA_theoretical_dispersion_curve(curve_t *curve){

    // First we compute the inverse of the wavelengths for the stiffness matrix computations:
    dfloat *k = (dfloat*) calloc(curve->curve_length, sizeof(dfloat));
    for (int i=0; i<curve->curve_length; ++i){
        k[i] = (2*M_PI) / curve->lambda_curve0[i]; // wavenumber is 2*pi/lambda
    }

    // Then we allocate the array of determinant values:
    dfloat *D = (dfloat*) calloc(curve->curve_length*curve->velocities_length, sizeof(dfloat));
    #define D(r, c) (D[(r)*curve->velocities_length + (c)])

    // Allocate the theoretically determined velocities and wavelengths:
    curve->c_t = (dfloat*) calloc(curve->curve_length, sizeof(dfloat));
    curve->lambda_t = (dfloat*) calloc(curve->curve_length, sizeof(dfloat));

    // For each entry in the dispersion curve-
    for (int l=0; l<curve->curve_length; ++l){
        // -we iterate over all test velocities-
        for (int m=0; m<curve->velocities_length; ++m){
            
            // -and compute their stiffness matrix determinants-
            D(l,m) = MASWA_stiffness_matrix(curve->c_test[m], k[l], curve->h,
                                               curve->alpha, curve->beta, curve->rho,
                                               curve->n);
            // -stopping when one has a sign change from its predecessor:
            if (m != 0 && D(l,m)*D(l,m-1) < 0.0){ // sign change indicated by (+) * (-) < 0
                curve->c_t[l] = curve->c_test[m];
                curve->lambda_t[l] = 2*M_PI / k[l]; // wavelength is 2*pi/wavenumber
                break;
            }
        }
    }

    free(D);
}
