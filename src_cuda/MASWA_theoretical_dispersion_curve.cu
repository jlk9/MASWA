#include "MASW.cuh"

/* Implements the theoretical dispersion curve in CUDA */


/* Helper to run the theoretical dispersion curve kernel */
void MASWA_theoretical_dispersion_curve_CUDA(curve_t *curve){
    
    /*
    Initialize stiffness matrices on kernel here
    */
    cuDoubleComplex **d_matrices;

    int size = 2*(curve->n+1);
    cuDoubleComplex     *d_data;
    cudaMalloc((void**)&d_matrices, curve->curve_length*curve->velocities_length*sizeof(cuDoubleComplex*));
    cudaMalloc(&d_data, size*size*curve->curve_length*curve->velocities_length*sizeof(cuDoubleComplex));
    kernel_assign_matrices_to_data<<<1,1>>>(d_matrices, d_data, curve->curve_length, curve->velocities_length, curve->n);

    MASWA_stiffness_matrix_CUDA(curve, d_matrices);

    const int blockSize = 256;
    int blocksPerWavelength = (curve->velocities_length / blockSize)+1;

    int *d_neighbors, *d_signChange;
    cudaMalloc(&d_neighbors, blocksPerWavelength*curve->curve_length*sizeof(int));
    cudaMalloc(&d_signChange, blocksPerWavelength*curve->curve_length*sizeof(int));

    dim3 numBlocks(curve->curve_length, blocksPerWavelength);

    kernel_block_sign_change<<<numBlocks, 256, 256>>>(d_neighbors, d_signChange, curve->velocities_length, size, d_matrices);

    dfloat *d_c_t, *d_c_test;
    cudaMalloc(&d_c_t, curve->curve_length*sizeof(dfloat));
    cudaMalloc(&d_c_test, curve->velocities_length*sizeof(dfloat));
    cudaMemcpy(d_c_test, curve->c_test, curve->velocities_length*sizeof(dfloat), cudaMemcpyHostToDevice);
    
    kernel_first_sign_change<<<1, 1>>>(d_c_t, d_c_test, d_signChange, blocksPerWavelength, curve->curve_length);
    cudaMemcpy(curve->c_t, d_c_t, curve->curve_length*sizeof(dfloat), cudaMemcpyDeviceToHost);

    // Compute the misfit
    int misfitBlocks = (curve->curve_length + blockSize - 1) / blockSize;

    dfloat *error, *d_error, *d_c_curve0;
    error = (dfloat*) calloc(1, sizeof(dfloat));
    cudaMalloc(&d_error, misfitBlocks*sizeof(dfloat));
    cudaMalloc(&d_c_curve0, curve->curve_length*sizeof(dfloat));
    cudaMemcpy(d_c_curve0, curve->c_curve0, curve->curve_length*sizeof(dfloat), cudaMemcpyHostToDevice);

    kernel_misfit_00<<<1, blockSize, blockSize>>>(curve->curve_length, d_c_t, d_c_curve0, d_error);

    cudaMemcpy(error, d_error, sizeof(dfloat), cudaMemcpyDeviceToHost);

    error[0] = (error[0]*100.0) / (dfloat) curve->curve_length;

    printf("error is %f\n", error[0]);

    cudaFree(d_data);
    cudaFree(d_matrices);
    cudaFree(d_neighbors);
    cudaFree(d_signChange);
    cudaFree(d_c_t);
    cudaFree(d_c_test);
    cudaFree(d_error);
    cudaFree(d_c_curve0);
    
}

/* Identifies the first dispersion curve with a sign change in each block of velocity values */
__global__ void kernel_block_sign_change(int *neighbors, int *signChange, int velocities_length, int size, cuDoubleComplex **matrices){
    /*
    IDEA: break up threads/blocks by c_test and lambda_curve0

    Each thread fills in one stiffness matrix corresponding to its lambda_curve0 and c_test values (thus we can parallelize ke_layer and ke_halfspace to fill these in)

    Compute determinants of heptadiagonal, symmetric stiffness matrices

    */

    // Both signChange and neighbors are size curve_length * blocksPerWavelength

    static const int blockSize = 256;
    int threadIndex = threadIdx.x;
    // We get the thread's matching wavelength value, which is its block id in the x axis:
    int wavelengthIndex = blockIdx.x;
    // Then we get its velocity value, which is determined by its thread id and block id in the y axis:
    int blockIndexY = blockIdx.y;
    int velocityIndex = blockIndexY * blockSize + threadIndex;
    int blocksPerWavelength = gridDim.y;

    // The portion of determinants stored in this block.
    // Since only the sign changes matter, they are stored as ints.
    __shared__ dfloat e[blockSize];

    e[threadIndex] = 0;
    if (velocityIndex < velocities_length){
        
        //This is where we get the determinants from the reduced matrices:
        cuDoubleComplex det = matrices[wavelengthIndex*velocities_length + velocityIndex][0];
        for (int i=1; i<size; ++i){
            det = cuCmul(matrices[wavelengthIndex*velocities_length + velocityIndex][i*size + i], det);
        }

        e[threadIndex] = cuCreal(det);
    }

    __syncthreads();

    #define neighbors(r, c) (neighbors[(r)*blocksPerWavelength + (c)])
    #define signChange(r, c) (signChange[(r)*blocksPerWavelength + (c)])
    if (threadIndex == 0){
        // Need the next nearest neighbor for sign change comparison, so the first entry of each block is stored in global memory.
        neighbors(wavelengthIndex, blockIndexY) = e[threadIndex];
        signChange(wavelengthIndex, blockIndexY) = -1;

        for (int i=0; i<blockSize-1; ++i){
            // If we find a sign change, we set the first sign change in this range of test velocities to that index and break:
            if (e[i]*e[i+1] < 0){
                signChange(wavelengthIndex, blockIndexY) = blockIndexY*blockSize + i + 1;
                break;
            }
        }

        
        // If we don't, we then check the first entry in the next range of test velocities for a sign change:
        if (signChange(wavelengthIndex, blockIndexY) == -1 && blockIndexY != blocksPerWavelength-1
            && neighbors(wavelengthIndex, blockIndexY+1)*e[blockSize-1] < 0){

            signChange(wavelengthIndex, blockIndexY) = blockIndexY*(blockSize+1);
        }
        
    }

}

/* Searches through array to find the first sign change for each wavelength.
    This iterates over multiple blocks from the previous kernel: 
*/
__global__ void kernel_first_sign_change(dfloat *c_t, dfloat* c_test, int *signChange, int blocksPerWavelength, int curve_length){

    #define signChange(r, c) (signChange[(r)*blocksPerWavelength + (c)])

    for (int i=0; i<curve_length; ++i){

        for (int j=0; j<blocksPerWavelength; ++j){
            if (signChange(i,j) != -1){
                c_t[i] =  c_test[signChange(i,j)];
                break;
            }
        }
    }

}




