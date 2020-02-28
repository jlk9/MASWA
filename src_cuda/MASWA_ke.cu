#include "MASW.cuh"

// Written by Joseph Kump (josek97@vt.edu). Last modified 12/20/2019

/*
    Fills in entries of all the stiffness matrices for each wavelength and test velocity,
    breaking up the task so each thread (across multiple blocks) sets up one stiffness
    matrix.
    
    Inputs:
    c_test              the array of test velocities
    lambda              the array of wavelengths in the dispersion curve
    h                   the thicknesses of the model layers
    alpha               the compressional wave velocities
    beta                the shear wave velocities
    rho                 the layer densities
    n                   the number of finite thickness layers in the model
    velocities_length   the length of c_test
    curve_length        the length of lambda (and the dispersion curve)
    matrices            the 2D array that stores the entries of all the stiffness matrices,
                        size 2(n+1) x 2(n+1)
*/
__global__ void kernel_generate_stiffness_matrices(dfloat *c_test, dfloat *lambda, dfloat *h, dfloat *alpha, dfloat *beta, dfloat *rho, int n, int velocities_length, int curve_length, cuDoubleComplex **matrices){

    // Each block is assigned to a wavelength, with each thread assigned to one velocity
    // and its stiffness matrix. Since there are usually more test velocities than threads
    // per block, each wavelength's matrices can be broken up over multiple blocks.
    
    int blockSize = blockDim.x;
    int threadIndex = threadIdx.x;
    int index = blockSize * blockIdx.x + threadIndex;
    
    // In case there are somehow more stiffness matrices than threads in the grid (very unlikely):
    int stride = blockSize * gridDim.x;
    
    // Although each Ke layer has 16 entries, the layers are symmetric and some of the 
    // entries are duplicates. So each stiffness matrix gets 6 entries for its Ke layers:
    int indexKe = threadIndex*6;

    // So the Ke layers can be accessed and incremented quickly:
    extern __shared__ cuDoubleComplex Ke[];

    #define matrices(i,j,k) (matrices[i][j*2*(n+1) + k])

    // Again, in the incredibly unlikely instance each thread has to fill in multiple matrices:
    for (int i=index; i<curve_length*velocities_length; i+=stride){

        dfloat k = 2*M_PI / lambda[i / velocities_length];

        for (int j=0; j<n; ++j){

            // Get a Ke layer for each finite thickness layer:
            kernel_Ke_layer_symm(h[j], alpha[j], beta[j], rho[j], c_test[i % velocities_length], k, Ke+indexKe);
            
            // Since the Ke layer entries do not line up perfectly with the stiffness matrix,
            // they need to be added separately (no option for a for loop here):
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

            // Stiffness matrix is symmetric:
            for (int r=1; r<4; ++r){
                for (int c=0; c<r; ++c){
                    matrices(i,(2*j+r),(2*j+c)) = matrices(i,(2*j+c),(2*j+r));
                }
            }
        }

        // Add the halfspace of the bottom layer to the bottom right 4 entries:
        kernel_Ke_halfspace_symm(alpha[n], beta[n], rho[n], c_test[i % velocities_length], k, Ke+indexKe);
        matrices(i, (2*n), (2*n))        = cuCadd(matrices(i, (2*n), (2*n)), Ke[indexKe]);
        matrices(i, (2*n), (2*n+1))      = cuCadd(matrices(i, (2*n), (2*n+1)), Ke[indexKe + 1]);
        matrices(i, (2*n+1), (2*n))      = matrices(i, (2*n), (2*n+1));
        matrices(i, (2*n+1), (2*n+1))    = cuCadd(matrices(i, (2*n+1), (2*n+1)), Ke[indexKe + 2]);
    }    
    
}

/* Modifies our test velocities so they are not too close to alpha or beta, causing
    numerical problems (MASWaves does this too). Parallel over one block, with the test
    velocities being divided over blocks.
    
    Inputs:
    velocities length   the length of c_test
    nPlus               the number of finite thickness layers + 1 (length of alpha and beta)
    c_test              the array of test velocities
    alpha               the array of compressional wave velocities in the model
    beta                the array of shear wave velocities in the model
    epsilon             the tolerance for how close any velocity in c_test can be to alpha or beta
*/
__global__ void kernel_too_close(int velocities_length, int nPlus, dfloat *c_test, dfloat *alpha, dfloat *beta, dfloat epsilon){

    int index = threadIdx.x;
    int blockSize = blockDim.x;

    // Break up all the test velocities over a block. Testing showed breaking this up over
    // multiple blocks had a negligible speed difference.
    for (int c=index; c<velocities_length; c+=blockSize){

        // Go through each model velocity, and increment the c_test entry if it is too close:
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
    
    Inputs:
    r_h         the thickness of the layer corresponding to this Ke layer
    r_alpha     the compressional wave velocity of this layer
    r_beta      the shear wave velocity of this layer
    r_rho       the density of this layer
    r_c_test    the test velocity of the stiffness matrix for this Ke layer
    r_k         the inverted wavelength of the stiffness matrix for this Ke layer
    Ke          the array where this layer is stored
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

    // Generate r and s as complex CUDA doubles:
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

    // Precompute parts of the layer entries:
    cuDoubleComplex rProduct = cuCmul(cuCmul(k,r),h);
    cuDoubleComplex sProduct = cuCmul(cuCmul(k,s),h);
    
    // Since the trig functions in CUDA do not accept complex parameters, we use trig
    // indenties to get the values instead:
    cuDoubleComplex Cr = make_cuDoubleComplex(cosh(cuCreal(rProduct))*cos(cuCimag(rProduct)), sinh(cuCreal(rProduct))*sin(cuCimag(rProduct)));
    cuDoubleComplex Sr = make_cuDoubleComplex(sinh(cuCreal(rProduct))*cos(cuCimag(rProduct)), cosh(cuCreal(rProduct))*sin(cuCimag(rProduct)));
    cuDoubleComplex Cs = make_cuDoubleComplex(cosh(cuCreal(sProduct))*cos(cuCimag(sProduct)), sinh(cuCreal(sProduct))*sin(cuCimag(sProduct)));
    cuDoubleComplex Ss = make_cuDoubleComplex(sinh(cuCreal(sProduct))*cos(cuCimag(sProduct)), cosh(cuCreal(sProduct))*sin(cuCimag(sProduct)));
    
    cuDoubleComplex One = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex neg = make_cuDoubleComplex(-1.0, 0.0);
    cuDoubleComplex Two = make_cuDoubleComplex(2.0, 0.0);

    cuDoubleComplex D = cuCadd(cuCmul(Two, cuCsub(One,cuCmul(Cr,Cs))), cuCmul(cuCmul(cuCadd(cuCdiv(One,cuCmul(r,s)), cuCmul(r,s)), Sr), Ss));

    // Now we make the 4x4 matrix as a 1D array and fill it in. The layer is symmetric and
    // has other redundant entries, so we only need 6 numbers:

    cuDoubleComplex krcd = cuCdiv(cuCmul(cuCmul(cuCmul(k,rho),c_test),c_test), D);

    Ke[0]   =   cuCmul(krcd, cuCsub(cuCdiv(cuCmul(Cr,Ss),s), cuCmul(r,cuCmul(Sr,Cs))));
    Ke[1]   =   cuCsub(cuCmul(krcd, cuCsub(cuCsub(cuCmul(Cr,Cs), cuCmul(cuCmul(r,s),cuCmul(Sr,Ss))), One)), cuCmul(k,cuCmul(rho,cuCmul(beta,cuCmul(beta,cuCadd(One,cuCmul(s,s)))))));
    Ke[2]   =   cuCmul(krcd, cuCsub(cuCmul(r,Sr), cuCdiv(Ss,s)));
    Ke[3]   =   cuCmul(krcd, cuCsub(Cs,Cr));
    
    Ke[4]   =   cuCmul(krcd, cuCsub(cuCdiv(cuCmul(Sr,Cs),r), cuCmul(cuCmul(s,Cr),Ss)));
    Ke[5]   =   cuCmul(krcd, cuCsub(cuCmul(s,Ss), cuCdiv(Sr,r)));
}

/* Creates the information for the final Ke halfspace, using CUDA complex numbers.
    Note Ke is length 3 since it is symmetric, although the halfspace actually has 4 entries.
    
    Inputs:
    r_alpha     the compressional wave velocity of the halfspace
    r_beta      the shear wave velocity of the halfspace
    r_rho       the density of the halfspace
    r_c_test    the test velocity of the stiffness matrix for this halfspace
    r_k         the inverted wavelength of the stiffness matrix for this halfspace
    Ke          the array where this halfspace is stored
    
*/
__device__ void kernel_Ke_halfspace_symm(dfloat r_alpha, dfloat r_beta, dfloat r_rho, dfloat r_c_test, dfloat r_k, cuDoubleComplex *Ke){

    // Similar tricks to Ke_layer, making sure r and s are properly initialized as complex
    // numbers and precomputing repetitive parts of the halfspace:
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

