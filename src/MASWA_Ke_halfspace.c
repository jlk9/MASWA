#include "MASW.h"

// Written by Joseph Kump (josek97@vt.edu). Last modified 09/23/2019

/* Computes the element stiffness matrix for the half-space (layer n+1) of the stratified earth model that is
 used in the inversion analysis. This halfspace is a 2x2 matrix stored as a 1D array by rows.
 
 Inputs:
 r_alpha    is an entry of the alpha array in a curve struct.
 r_beta     is an entry of the beta array in a curve struct.
 r_rho      is an entry of the rho array in a curve struct.
 r_c_test   is a test velocity from a curve struct.
 r_k        is an inverted wavelength from a curve struct.
 
 Output:
 Ke         the 2x2 matrix representing the infinite-thickness layer at the bottom of the
                proposed ground model, which increments the stiffness matrix
 
*/
compfloat *MASWA_Ke_halfspace(dfloat r_alpha, dfloat r_beta, dfloat r_rho, dfloat r_c_test, dfloat r_k){

    compfloat alpha     = r_alpha;
    compfloat beta      = r_beta;
    compfloat rho       = r_rho;
    compfloat c_test    = r_c_test;
    compfloat k         = r_k;
    
    compfloat r    = sqrt(1.0 - (c_test*c_test)/(alpha*alpha));
    compfloat s    = sqrt(1.0 - (c_test*c_test)/(beta*beta));

    /* Complex types in c++ can have both +0 and -0. sqrt(x - 0i) will
        be a negative imaginary number, which we don't want. If either r
        or s is imaginary, we want it to be positive.
    */
    if (imag(r) < 0){
        r *= -1.0;
    }
    if (imag(s) < 0){
        s *= -1.0;
    }
    
    // Once we get r and s, we can compute the entries of the halfspace:
    compfloat *Ke  = (compfloat*) calloc(4, sizeof(compfloat));
    compfloat temp = k*rho*beta*beta*(1.0-s*s)/(1.0-r*s);

    Ke[0] = r*temp;
    Ke[1] = temp - 2.0*k*rho*beta*beta;
    Ke[2] = Ke[1];
    Ke[3] = s*temp;

    return Ke;
}

