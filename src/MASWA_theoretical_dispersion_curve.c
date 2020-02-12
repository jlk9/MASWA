#include "MASW.h"


/* Computes the fundamental mode theoretical dispersion curve. This is done by using the stiffness matrix method,
    which is also used in MASWaves.
*/
void MASWA_theoretical_dispersion_curve(curve_t *curve){

    dfloat *k = (dfloat*) calloc(curve->curve_length, sizeof(dfloat));
    for (int i=0; i<curve->curve_length; ++i){
        k[i] = (2*M_PI) / curve->lambda_curve0[i];
    }

    dfloat *D = (dfloat*) calloc(curve->curve_length*curve->velocities_length, sizeof(dfloat));
    #define D(r, c) (D[(r)*curve->velocities_length + (c)])

    curve->c_t = (dfloat*) calloc(curve->curve_length, sizeof(dfloat));
    curve->lambda_t = (dfloat*) calloc(curve->curve_length, sizeof(dfloat));

    for (int l=0; l<curve->curve_length; ++l){
        for (int m=0; m<curve->velocities_length; ++m){
            
            D(l,m) = MASWA_stiffness_matrix(curve->c_test[m], k[l], curve->h,
                                               curve->alpha, curve->beta, curve->rho,
                                               curve->n);
            if (m != 0 && D(l,m)*D(l,m-1) < 0.0){
                curve->c_t[l] = curve->c_test[m];
                curve->lambda_t[l] = 2*M_PI / k[l];
                break;
            }
        }
    }

    free(D);
}
