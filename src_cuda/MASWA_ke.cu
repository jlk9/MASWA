#include "MASW.cuh"

/* The function MASWaves_Ke_layer computes the element stiffness matrix
 of the j-th layer (j = 1,...,n) of the stratified earth
 model that is used in the inversion analysis. The stiffness matrix as a
 4x4 stored in a 1D Array. */

/*
    Fills in entries of stiffness matrices utilizing multiple blocks, and takes advantage of the symmetry in Ke_layer and Ke_halfspace
*/
__global__ void kernel_generate_stiffness_matrices(dfloat *c_test, dfloat *lambda, dfloat *h, dfloat *alpha, dfloat *beta, dfloat *rho, int n, int velocities_length, int curve_length, cuDoubleComplex **matrices){

    int blockSize = blockDim.x;
    int threadIndex = threadIdx.x;
    int index = blockSize * blockIdx.x + threadIndex;
    int stride = blockSize * gridDim.x;
    int indexKe = threadIndex*6;

    extern __shared__ cuDoubleComplex Ke[];

    #define matrices(i,j,k) (matrices[i][j*2*(n+1) + k])

    for (int i=index; i<curve_length*velocities_length; i+=stride){

        dfloat k = 2*M_PI / lambda[i / velocities_length];

        for (int j=0; j<n; ++j){

            kernel_Ke_layer_symm(h[j], alpha[j], beta[j], rho[j], c_test[i % velocities_length], k, Ke+indexKe);
            
            matrices(i,(2*j),(2*j))     = cuCadd(matrices(i,(2*j),(2*j)), Ke[indexKe]);
            matrices(i,(2*j),(2*j+1))   = cuCadd(matrices(i,(2*j),(2*j+1)), Ke[indexKe + 1]);
            matrices(i,(2*j),(2*j+2))   = cuCadd(matrices(i,(2*j),(2*j+2)), Ke[indexKe + 2]);
            matrices(i,(2*j),(2*j+3))   = cuCadd(matrices(i,(2*j),(2*j+3)), Ke[indexKe + 3]);
            matrices(i,(2*j+1),(2*j+1)) = cuCadd(matrices(i,(2*j+1),(2*j+1)), Ke[indexKe + 4]);
            matrices(i,(2*j+1),(2*j+2)) = cuCsub(matrices(i,(2*j+1),(2*j+2)), Ke[indexKe + 3]);
            matrices(i,(2*j+1),(2*j+3)) = cuCadd(matrices(i,(2*j+1),(2*j+3)), Ke[indexKe + 5]);
            matrices(i,(2*j+2),(2*j+2)) = cuCadd(matrices(i,(2*j+2),(2*j+2)), Ke[indexKe]);
            matrices(i,(2*j+2),(2*j+3)) = cuCsub(matrices(i,(2*j+2),(2*j+3)), Ke[indexKe + 1]);
            matrices(i,(2*j+3),(2*j+3)) = cuCadd(matrices(i,(2*j+3),(2*j+3)), Ke[indexKe + 4]);

            for (int r=1; r<4; ++r){
                for (int c=0; c<r; ++c){
                    matrices(i,(2*j+r),(2*j+c)) = matrices(i,(2*j+c),(2*j+r));
                }
            }
        }

        kernel_Ke_halfspace_symm(alpha[n], beta[n], rho[n], c_test[i % velocities_length], k, Ke+indexKe);
        matrices(i, (2*n), (2*n))        = cuCadd(matrices(i, (2*n), (2*n)), Ke[indexKe]);
        matrices(i, (2*n), (2*n+1))      = cuCadd(matrices(i, (2*n), (2*n+1)), Ke[indexKe + 1]);
        matrices(i, (2*n+1), (2*n))      = matrices(i, (2*n), (2*n+1));
        matrices(i, (2*n+1), (2*n+1))    = cuCadd(matrices(i, (2*n+1), (2*n+1)), Ke[indexKe + 2]);
    }    
    
}

/* Modifies our test velocities so they are not too close to alpha or beta, causing numerical problems (MASWaves does this too).
    Parallel over one block.
*/
__global__ void kernel_too_close(int velocities_length, int nPlus, dfloat *c_test, dfloat *alpha, dfloat *beta, dfloat epsilon){

    int index = threadIdx.x;
    int blockSize = blockDim.x;

    for (int c=index; c<velocities_length; c+=blockSize){

        for (int i=0; i<nPlus; ++i){
            while (abs(c_test[c]-alpha[i]) < epsilon || abs(c_test[c]-beta[i]) < epsilon){
                c_test[c] *= 1-epsilon;
            }
        }
    }
}

/* Constructs a Ke Layer in a single CUDA thread, using CUDA complex numbers. Note Ke is length 6.
    This is because we have accounted for symmetry and other redundant entries in the Ke layer, reducing
    the memory requirements and number of computations.
*/
__device__ void kernel_Ke_layer_symm(dfloat r_h, dfloat r_alpha, dfloat r_beta, dfloat r_rho, dfloat r_c_test, dfloat r_k, cuDoubleComplex *Ke){

    // Idea: make rsquare and ssquare real, then take the square root of their absolute value.
    // Make them into complex numbers. If negative, then initialize them as imaginary, real if positive.
    // Use formulas for sinh,... of complex numbers

    cuDoubleComplex h       = make_cuDoubleComplex(r_h, 0.0);
    cuDoubleComplex alpha   = make_cuDoubleComplex(r_alpha, 0.0);
    cuDoubleComplex beta    = make_cuDoubleComplex(r_beta, 0.0);
    cuDoubleComplex rho     = make_cuDoubleComplex(r_rho, 0.0);
    cuDoubleComplex c_test  = make_cuDoubleComplex(r_c_test, 0.0);
    cuDoubleComplex k       = make_cuDoubleComplex(r_k, 0.0);

    cuDoubleComplex r, s;

    dfloat rSquare = 1.0 - (r_c_test*r_c_test)/(r_alpha*r_alpha);
    dfloat sSquare = 1.0 - (r_c_test*r_c_test)/(r_beta*r_beta);

    if (rSquare<0.0){
        r = make_cuDoubleComplex(0, sqrt(-1.0*rSquare));
    }
    else{
        r = make_cuDoubleComplex(sqrt(rSquare), 0);
    }
    if (sSquare<0.0){
        s = make_cuDoubleComplex(0, sqrt(-1.0*sSquare));
    }
    else{
        s = make_cuDoubleComplex(sqrt(sSquare), 0);
    }

    cuDoubleComplex rProduct = cuCmul(cuCmul(k,r),h);
    cuDoubleComplex sProduct = cuCmul(cuCmul(k,s),h);
    
    cuDoubleComplex Cr = make_cuDoubleComplex(cosh(cuCreal(rProduct))*cos(cuCimag(rProduct)), sinh(cuCreal(rProduct))*sin(cuCimag(rProduct)));
    cuDoubleComplex Sr = make_cuDoubleComplex(sinh(cuCreal(rProduct))*cos(cuCimag(rProduct)), cosh(cuCreal(rProduct))*sin(cuCimag(rProduct)));
    cuDoubleComplex Cs = make_cuDoubleComplex(cosh(cuCreal(sProduct))*cos(cuCimag(sProduct)), sinh(cuCreal(sProduct))*sin(cuCimag(sProduct)));
    cuDoubleComplex Ss = make_cuDoubleComplex(sinh(cuCreal(sProduct))*cos(cuCimag(sProduct)), cosh(cuCreal(sProduct))*sin(cuCimag(sProduct)));
    
    cuDoubleComplex One = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex neg = make_cuDoubleComplex(-1.0, 0.0);
    cuDoubleComplex Two = make_cuDoubleComplex(2.0, 0.0);

    cuDoubleComplex D = cuCadd(cuCmul(Two, cuCsub(One,cuCmul(Cr,Cs))), cuCmul(cuCmul(cuCadd(cuCdiv(One,cuCmul(r,s)), cuCmul(r,s)), Sr), Ss));

    // Now we make the 4x4 matrix as a 1D array and fill it in

    cuDoubleComplex krcd = cuCdiv(cuCmul(cuCmul(cuCmul(k,rho),c_test),c_test), D);

    Ke[0]   =   cuCmul(krcd, cuCsub(cuCdiv(cuCmul(Cr,Ss),s), cuCmul(r,cuCmul(Sr,Cs))));
    Ke[1]   =   cuCsub(cuCmul(krcd, cuCsub(cuCsub(cuCmul(Cr,Cs), cuCmul(cuCmul(r,s),cuCmul(Sr,Ss))), One)), cuCmul(k,cuCmul(rho,cuCmul(beta,cuCmul(beta,cuCadd(One,cuCmul(s,s)))))));
    Ke[2]   =   cuCmul(krcd, cuCsub(cuCmul(r,Sr), cuCdiv(Ss,s)));
    Ke[3]   =   cuCmul(krcd, cuCsub(Cs,Cr));
    
    Ke[4]   =   cuCmul(krcd, cuCsub(cuCdiv(cuCmul(Sr,Cs),r), cuCmul(cuCmul(s,Cr),Ss)));
    Ke[5]   =   cuCmul(krcd, cuCsub(cuCmul(s,Ss), cuCdiv(Sr,r)));
}

/* Creates the information for the final Ke halfspace, using cuda complex numbers.
    Note Ke is length 3 since it is symmetric in this case.
*/
__device__ void kernel_Ke_halfspace_symm(dfloat r_alpha, dfloat r_beta, dfloat r_rho, dfloat r_c_test, dfloat r_k, cuDoubleComplex *Ke){

    cuDoubleComplex alpha   = make_cuDoubleComplex(r_alpha, 0.0);
    cuDoubleComplex beta    = make_cuDoubleComplex(r_beta, 0.0);
    cuDoubleComplex rho     = make_cuDoubleComplex(r_rho, 0.0);
    cuDoubleComplex c_test  = make_cuDoubleComplex(r_c_test, 0.0);
    cuDoubleComplex k       = make_cuDoubleComplex(r_k, 0.0);

    dfloat rSquare = 1.0 - (r_c_test*r_c_test)/(r_alpha*r_alpha);
    dfloat sSquare = 1.0 - (r_c_test*r_c_test)/(r_beta*r_beta);

    cuDoubleComplex r, s;

    if (rSquare<0.0){
        r = make_cuDoubleComplex(0, sqrt(-1.0*rSquare));
    }
    else{
        r = make_cuDoubleComplex(sqrt(rSquare), 0);
    }
    if (sSquare<0.0){
        s = make_cuDoubleComplex(0, sqrt(-1.0*sSquare));
    }
    else{
        s = make_cuDoubleComplex(sqrt(sSquare), 0);
    }

    cuDoubleComplex One = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex Two = make_cuDoubleComplex(2.0, 0.0);
    cuDoubleComplex temp = cuCdiv(cuCmul(cuCmul(cuCmul(k,rho),cuCmul(beta,beta)),cuCsub(One,cuCmul(s,s))), cuCsub(One,cuCmul(r,s)));

    Ke[0] = cuCmul(temp, r);
    Ke[1] = cuCsub(temp, cuCmul(Two, cuCmul(cuCmul(k,rho),cuCmul(beta,beta))));
    Ke[2] = cuCmul(temp, s);
}

