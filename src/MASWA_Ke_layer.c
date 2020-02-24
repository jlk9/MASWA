#include "MASW.h"

/* The function MASWaves_Ke_layer computes the element stiffness matrix
 of the j-th layer (j = 1,...,n) of the stratified earth
 model that is used in the inversion analysis. The stiffness matrix is a
 4x4 stored in a 1D Array. */

compfloat *MASWA_Ke_layer(dfloat r_h, dfloat r_alpha, dfloat r_beta, dfloat r_rho, dfloat r_c_test, dfloat r_k){

    /* We need to convert each of the model parameters into complex numbers for the stiffness matrices,
        hence naming them "r_..." in the function call.
    */
    compfloat h = r_h;
    compfloat alpha = r_alpha;
    compfloat beta = r_beta;
    compfloat rho = r_rho;
    compfloat c_test = r_c_test;
    compfloat k = r_k;

    compfloat r = (compfloat) sqrt(1.0 - (c_test*c_test)/(alpha*alpha));
    compfloat s = (compfloat) sqrt(1.0 - (c_test*c_test)/(beta*beta));
    
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

    compfloat Cr = cosh(k*r*h);
    compfloat Sr = sinh(k*r*h);
    compfloat Cs = cosh(k*s*h);
    compfloat Ss = sinh(k*s*h);

    compfloat D = 2.0*(1.0-Cr*Cs) + (1.0/(r*s) + r*s)*Sr*Ss;

    compfloat *Ke = (compfloat*) calloc(16, sizeof(compfloat));
    compfloat krcd = (k*rho*c_test*c_test)/D;

    // Filling out the entries for the stiffness matrix. Rows have been broken up for ease of reading:
    Ke[0]   =   krcd * (Cr*Ss/s - r*Sr*Cs);
    Ke[1]   =   krcd * (Cr*Cs - r*s*Sr*Ss - 1.0) - k*rho*beta*beta*(1.0+s*s);
    Ke[2]   =   krcd * (r*Sr - Ss/s);
    Ke[3]   =   krcd * (Cs - Cr);
    
    Ke[4]   =   Ke[1];
    Ke[5]   =   krcd * (Sr*Cs/r - s*Cr*Ss);
    Ke[6]   =   -Ke[3];
    Ke[7]   =   krcd * (s*Ss - Sr/r);
    
    Ke[8]   =   Ke[2];
    Ke[9]   =   Ke[6];
    Ke[10]  =   Ke[0];
    Ke[11]  =   -Ke[1];
    
    Ke[12]  =   Ke[3];
    Ke[13]  =   Ke[7];
    Ke[14]  =   -Ke[1];
    Ke[15]  =   Ke[5];

    return Ke;
}
