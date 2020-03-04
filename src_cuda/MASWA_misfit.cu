#include "MASW.cuh"

// Written by Joseph Kump (josek97@vt.edu). Last modified 10/30/2019

/* Implements the misfit functions in CUDA */

/* Runs the kernel for computing the misfit. Not used for the inversion (since it's more efficient to
    just compute the misfit with the dispersion curves while they're already on the GPU), but still
    useful for testing purposes.

    Inputs:
    curve_length    the length of both dispersion curves
    c_t             the theoretically determined dispersion curve from MASW inversion
    c_curve0        the experimentally derived dispersion curve
    
    Outputs:
    returns the error for the multi-block misfit, but prints out both values and their
        runtimes
*/
dfloat run_misfit(int curve_length, dfloat *c_t, dfloat *c_curve0){

    // Allocate memory on the GPU for the dispersion curves and error, and transferring
    // the dispersion curves:
    dfloat *d_ct, *d_ccurve0, *d_error0, *d_error1, *error;

    error = (dfloat*) calloc(1, sizeof(dfloat));
    cudaMalloc(&d_ct, curve_length*sizeof(dfloat));
    cudaMalloc(&d_ccurve0, curve_length*sizeof(dfloat));
    cudaMalloc(&d_error0, sizeof(dfloat));

    cudaMemcpy(d_ct, c_t, curve_length*sizeof(dfloat), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ccurve0, c_curve0, curve_length*sizeof(dfloat), cudaMemcpyHostToDevice);

    int blockSize = 256;

    // First testing the speed of the single-block implementation:
    double time_spent0  = 0.0;
    clock_t begin0      = clock();

    kernel_misfit_00<<<1, blockSize, blockSize>>>(curve_length, d_ct, d_ccurve0, d_error0);

    clock_t end0    = clock();
	time_spent0    += (double)(end0 - begin0) / CLOCKS_PER_SEC;
    printf("Time for misfit with one block is %f\n", time_spent0);

    cudaMemcpy(error, d_error0, sizeof(dfloat), cudaMemcpyDeviceToHost);

    // Now testing the speed of the multi-block implementation:
    int numBlocks = (curve_length + blockSize - 1) / blockSize;
    cudaMalloc(&d_error1, numBlocks*sizeof(dfloat));

    double time_spent1  = 0.0;
    clock_t begin1      = clock();

    kernel_misfit_01<<<numBlocks, blockSize, blockSize>>>(curve_length, d_ct, d_ccurve0, d_error1);
    kernel_misfit_block_summation<<<1, blockSize, blockSize>>>(numBlocks, d_error1);

    clock_t end1    = clock();
	time_spent1    += (double)(end1 - begin1) / CLOCKS_PER_SEC;
    printf("Time for misfit with multiple blocks is %f\n", time_spent1);

    cudaMemcpy(error, &d_error1[0], sizeof(dfloat), cudaMemcpyDeviceToHost);

    cudaFree(d_ct);
    cudaFree(d_ccurve0);
    cudaFree(d_error0);
    cudaFree(d_error1);

    return error[0];
}

/* Global kernel, computes the misfit between the real and theoretical curves. Since the dispersion curve
    is typically small, it is usually more efficient to compute the misfit within a single block.

    Inputs:
    curve_length    the length of both dispersion curves
    c_t             the theoretically determined dispersion curve from MASW inversion
    c_curve0        the experimentally derived dispersion curve
    error           the total misfit across the dispersion curve
    
    Output:
        void, but sets the error value in the error parameter
*/
__global__ void kernel_misfit_00(const int curve_length, dfloat *c_t, dfloat *c_curve0, dfloat *error){

    static const int blockSize = 256; //We're treating blocksize as 256 for now

    // Use the index to determine which entries of the dispersion curve each thread handles,
    // and the stride if the dispersion curve is longer than the block size:
    int index   = threadIdx.x;
    int stride  = blockSize;

    // Stores element errors in shared memory:
    __shared__ dfloat e[blockSize];

    e[index] = 0.0;

    // For each entry of the dispersion curve handled by this thread, compute the error and
    // add it to shared memory:
    for (int i=index; i<curve_length; i+=stride){
        e[index] += sqrt((c_curve0[i]-c_t[i])*(c_curve0[i]-c_t[i])) / c_curve0[i];
    }

    // Here the errors from all the threads are summed up, avoiding race conditions:
    __syncthreads();
    for (int size = stride/2; size>0; size/=2) { //uniform
        if (index<size)
            e[index] += e[index+size];
        __syncthreads();
    }
    error[0] = e[0];

}

/* Global kernel, computes the misfit between the real and theoretical curves over multiple blocks. Need
    to use kernel_misfit_block_summation below to sum up these block misfits. Generally the single block
    implementation is more efficient, so that is used by default.

    Inputs:
    curve_length    the length of both dispersion curves
    c_t             the theoretically determined dispersion curve from MASW inversion
    c_curve0        the experimentally derived dispersion curve
    error           the total misfit across the dispersion curve, split by blocks
    
    Outputs:
    void, but sets the errors in the error array (each entry contains the summed errors
        of one block).
*/
__global__ void kernel_misfit_01(const int curve_length, dfloat *c_t, dfloat *c_curve0, dfloat *error){


    // Generally the same as the 1-block function, except the summation at the end puts all
    // the errors in one block into one entry. (So error here is an array of numBlock entries,
    // where it is length 1 in the 1-block function)
    static const int blockSize = 256; //We're treating blocksize as 256 for now

    int threadIndex     = threadIdx.x;
    int gridIndex       = threadIndex + blockIdx.x*blockSize;
    int stride          = blockSize * gridDim.x;

    __shared__ dfloat e[blockSize];

    e[threadIndex] = 0.0;

    for (int i=gridIndex; i<curve_length; i+=stride){
        e[threadIndex] += sqrt((c_curve0[i]-c_t[i])*(c_curve0[i]-c_t[i])) / c_curve0[i];
    }

    // Reduce within the blocks:
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { //uniform
        if (threadIndex<size)
            e[threadIndex] += e[threadIndex+size];
        __syncthreads();
    }

    if (threadIndex == 0){
        error[blockIdx.x] = e[0];
    }
}

/* Sums all the entries in an array and stores them in the first entry. This is used to add up
    multiple blocks' worth of memory for the multi-block misfit.

    Inputs:
    array_length    the length of the array to sum up
    array           the array whose entries will be summed up
    
    Outputs:
    void, but stores the sum of all the array's entries in its first entry
*/
__global__ void kernel_misfit_block_summation(const int array_length, dfloat *array){

    static const int blockSize  = 256;
    int index                   = threadIdx.x;
    int stride                  = blockSize;

    // Copy the entries into shared memory for faster access:
    __shared__ dfloat e[blockSize];
    e[index] = 0.0;

    for (int i=index; i<array_length; i+=stride){
        e[index] += array[i];
    }

    // Now add up the entries in shared memory in a reduce-style operation:
    __syncthreads();
    for (int size = stride/2; size>0; size/=2) { //uniform
        if (index<size)
            e[index] += e[index+size];
        __syncthreads();
    }
    array[0] = e[0];

}

