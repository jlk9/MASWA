#include "MASW.cuh"

// Written by Joseph Kump (josek97@vt.edu). Last modified 12/20/2019

#define matrices(i,j,k) (matrices[i][j*size + k])

/* Generates the stiffness matrices and fills in their entries, then uses Gaussian
    elimination to make finding their determinants easy. Mostly just allocates memory on
    the GPU, then gets GPU kernels to do the heavy lifting.
    
    Inputs:
    curve       the dispersion curve struct
    d_matrices  the stiffness matrices allocated on the GPU as a 2D array (each sub-array
                    corresponding to a matrix). These were already allocated.
*/
void MASWA_stiffness_matrix_CUDA(curve_t *curve, cuDoubleComplex **d_matrices){

    // The matrix axes are based on the number of layers. Then the model parameters, test
    // velocities, and dispersion curve wavelengths are allocated and transferred from
    // the CPU:
    int size = 2*(curve->n+1);
    dfloat *d_c_test, *d_lambda, *d_h, *d_alpha, *d_beta, *d_rho;

    cudaMalloc(&d_c_test, curve->velocities_length*sizeof(dfloat));
    cudaMalloc(&d_lambda, curve->curve_length*sizeof(dfloat));
    cudaMalloc(&d_h, curve->n*sizeof(dfloat));
    cudaMalloc(&d_alpha, (curve->n+1)*sizeof(dfloat));
    cudaMalloc(&d_beta, (curve->n+1)*sizeof(dfloat));
    cudaMalloc(&d_rho, (curve->n+1)*sizeof(dfloat));

    cudaMemcpy(d_c_test, curve->c_test, curve->velocities_length*sizeof(dfloat), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lambda, curve->lambda_curve0, curve->curve_length*sizeof(dfloat), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, curve->h, curve->n*sizeof(dfloat), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, curve->alpha, (curve->n+1)*sizeof(dfloat), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, curve->beta, (curve->n+1)*sizeof(dfloat), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, curve->rho, (curve->n+1)*sizeof(dfloat), cudaMemcpyHostToDevice);

    // If the shared memory requirements for kernel_generate_stiffness_matrices are too large, then this can be reduced to 128:
    int blockSize   = 256;
    int blocks      = (curve->curve_length*curve->velocities_length / blockSize)+1;
    
    // Check none of the test velocities are too close to the model velocities:
    kernel_too_close<<<1,blockSize>>>(curve->velocities_length, curve->n+1, d_c_test, d_alpha, d_beta, 0.0001);

    // Fill in the stiffness matrices here:
    kernel_generate_stiffness_matrices<<<blocks, blockSize, 6*blockSize*sizeof(cuDoubleComplex)>>>(d_c_test, d_lambda, d_h, d_alpha, d_beta, d_rho, curve->n, curve->velocities_length, curve->curve_length, d_matrices);
    
    // Gaussian Elimination here:
    kernel_hepta_determinant_CUDA<<<blocks, blockSize, 4*blockSize*sizeof(cuDoubleComplex)>>>(curve->curve_length, curve->velocities_length, size, d_matrices);

    // Free all of the model parameters, since they are no longer necessary:
    cudaFree(d_c_test);
    cudaFree(d_lambda);
    cudaFree(d_h);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_rho);

}

/* Fills in a simple identity matrix for some test cases. Not used as part of MASW (though
    the parameters are modelled after it).

    Inputs:
    velocities_length       determines the number of matrices
    curve_length            also determines the number of matrices
    n                       determines the size of the matrices
    matrices                the array storing the matrix entries
*/
__global__ void kernel_matrix_fill_in_serial(int velocities_length, int curve_length, int n, cuDoubleComplex **matrices){

    int size = 2*(n+1);
    

    for (int i=0; i<curve_length*velocities_length; ++i){
        for (int j=0; j<size; ++j){
            matrices(i,j,j) = make_cuDoubleComplex(1, 0);
        }
    }
}

/* Assigns matrix pointers to the contiguous chunk of memory where they are stored. This is necessary for the implementation
    utilizing cuBLAS (there may be a method that does not require a kernel, but other things I tried generated a seg fault).
    Since cuBLAS is no longer used, the code may be rewritten to omit this. (Matrices can just be stored as a 1D array without
    pointers, as opposed to a 2D array).
*/
__global__ void kernel_assign_matrices_to_data(cuDoubleComplex **matrices, cuDoubleComplex *data, int curve_length, int velocities_length, int n){

    // Currently this kernel is serial, but it's effect on the runtime is negligible.
    // Could be parallelized, or the code could be written so the GPU matrices are just all
    // on a big 1D array:
    for (int i=0; i<curve_length*velocities_length; ++i){

        matrices[i] = (cuDoubleComplex*) ((char*) data+i*((size_t)(4*n*n + 8*n + 4))*sizeof(cuDoubleComplex));
    }
}

/* Performs Gaussian elimination on the stiffness matrices by taking advantage of their
    banded structure, unlike in the cuBLAS function. This version puts the current row
    used for elimination into shared memory, improving the speed of accessing it.
    
    Inputs:
        curve_length            the length of the dispersion curve
        velocities_length       the number of test velocities
        size                    the length of the axis for each stiffness matrix
        matrices                the 2D array holding the matrices' entries
*/
__global__ void kernel_hepta_determinant_CUDA(int curve_length, int velocities_length, int size, cuDoubleComplex **matrices){

    // The block and thread indices are used to allocate stiffness matrices to each thread:
    int blockSize   = blockDim.x;
    int threadIndex = threadIdx.x;
    int index       = blockSize * blockIdx.x + threadIndex;
    int stride      = blockSize * gridDim.x;

    // Each row used for reduction has only 4 nonzero entries, so we only need 4 entries
    // in shared memory for each matrix:
    int sharedIndex = 4*threadIndex;

    extern __shared__ cuDoubleComplex row[];

    for (int x=index; x<curve_length*velocities_length; x+=stride){

        dfloat modifier = 1.0;

        for (int i=0; i<size; ++i){

            int end = i + 4;
            if (end > size){
                end = size;
            }

            for (int k=i; k<end; ++k){
                row[sharedIndex + k - i] = matrices(x,i,k);
            }

            // Row switching
            if (cuCreal(matrices(x,i,i)) == 0.0 && cuCimag(matrices(x,i,i)) == 0.0){

                for (int s=i+1; s<end; ++s){
                    if(cuCreal(matrices(x,s,i)) != 0.0 || cuCimag(matrices(x,s,i)) != 0.0){
                        cuDoubleComplex temp;
                        for (int w=i; w<end; ++w){
                            temp = matrices(x,i,w);
                            matrices(x,i,w) = matrices(x,s,w);
                            matrices(x,s,w) = temp;
                            modifier *= -1;
                            break;
                        }
                    }
                }
            }

            //Gaussian elimination for the three rows (or fewer) below this one:
            for (int j=i+1; j<end; ++j){

                cuDoubleComplex coeff = cuCdiv(matrices(x,j,i), row[sharedIndex]);

                for (int k=i+1; k<end; ++k){

                    matrices(x,j,k) = cuCsub(matrices(x,j,k), cuCmul(coeff, row[sharedIndex + k - i]));
                }
            }
        }

        // account for determinant changing from row switching
        matrices(x,0,0) = cuCmul(matrices(x,0,0), make_cuDoubleComplex(modifier, 0.0));
    }

}








