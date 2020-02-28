#include "MASW.cuh"

// Written by Joseph Kump (josek97@vt.edu). Last modified 01/27/2020

/*
Suite of test cases for various CUDA functions.
*/

/* End-to-end test for the CUDA implementation of MASW. */
int testProcess(){

    const int n = 6;
    dfloat h[n] = {1.0, 1.0, 2.0, 2.0, 4.0, 5.0};
    dfloat alpha[n+1] = {1440.0, 1440.0, 1440.0, 1440.0, 1440.0, 1440.0, 1440.0};
    dfloat beta[n+1] = {75.0, 90.0, 150.0, 180.0, 240.0, 290.0, 290.0};
    dfloat rho[n+1] = {1850.0, 1850.0, 1850.0, 1850.0, 1850.0, 1850.0, 1850.0};

    const int c_length = 1001;
    const int l_length = 40;
    
    //60.0000, 35.3333 are first 2
    dfloat lambda[l_length] = {28.4400, 25.3091, 22.0000, 18.9231, 17.0571, 14.9600, 13.2000,
                           11.5059, 10.0667, 8.5263, 7.5000, 6.7429, 6.2727, 5.6870, 5.2000, 4.8480, 4.5231,
                            4.1333, 3.9000, 3.7241, 3.5200, 3.3677, 3.2250, 3.0909, 2.9647, 2.8457, 2.7333,
                            2.6270, 2.4947, 2.4000, 2.3100, 2.2537, 2.2000, 2.1209, 2.0727, 2.0000, 1.9304,
                            1.8638, 1.8250, 1.7878};
    
    dfloat c_curve[l_length] = {237.0, 232.0, 220.0, 205.0, 199.0, 187.0, 176.0, 163.0,
                                 151.0, 135.0, 125.0, 118.0, 115.0, 109.0, 104.0, 101.0,
                                 98.0, 93.0, 91.0, 90.0, 88.0, 87.0, 86.0, 85.0, 84.0,
                                 83.0, 82.0, 81.0, 79.0, 78.0, 77.0, 77.0, 77.0, 76.0, 
                                 76.0, 75.0, 74.0, 73.0, 73.0, 73.0};

    dfloat *c_array = (dfloat*) calloc(c_length, sizeof(dfloat));
    dfloat value = 0.5;
    
    for (int i=0; i<c_length; ++i){
        c_array[i] = value;
        value += 0.5;
    }
    
    curve_t *curve = (curve_t*) calloc(1, sizeof(curve_t));
    
    MASWA_setVariables(curve, c_array, h, alpha, beta, rho, n, c_length, l_length, c_curve, lambda);
    curve->c_test = c_array;

    MASWA_inversion_CUDA(curve);

    for (int i=0; i<curve->curve_length; ++i){
        printf("%f, ", curve->c_t[i]);
    }
    printf("\n\n");

    return 0;
}

/* Runs testProcess 11 times (accounting for potential compilation time in first run).
*/
int testProcess_full(){
    int result = 0;
    int trials = 10;
    double *time_spent = (double*) calloc(trials+1, sizeof(double));
    clock_t begin, end;

    for (int i=0; i<trials+1; ++i){
        begin = clock();
        result = testProcess();
        end = clock();
        time_spent[i] = (double)(end - begin) / CLOCKS_PER_SEC;
    }
    for(int i=0; i<trials+1; ++i){
        printf("%f,", time_spent[i]);
    }
    printf("\n\n");

    return result;
}

/* Tests the sign change across multiple blocks works properly. The vector c_t should return [0, 2, 3, 4, 5]
*/
int testSignChange(){
    int blocksPerWavelength = 3;
    int curve_length = 5;

    dfloat *c_t = (dfloat*) calloc(curve_length, sizeof(dfloat));
    dfloat *c_test = (dfloat*) calloc(10, sizeof(dfloat));
    for (int i=0; i<10; ++i){
        c_test[i] = i+1;
    }

    int *signChange = (int*) calloc(blocksPerWavelength*curve_length, sizeof(int));
    // Case 1: no sign change
    signChange[0] = -1;
    signChange[1] = -1;
    signChange[2] = -1;
    // Case 2: 1st entry is a sign change
    signChange[3] = 1;
    signChange[4] = -1;
    signChange[5] = -1;
    // Case 3: 2nd entry is the one
    signChange[6] = -1;
    signChange[7] = 2;
    signChange[8] = -1;
    // Case 4: 3rd entry
    signChange[9] = -1;
    signChange[10] = -1;
    signChange[11] = 3;
    // Case 5: All 3 entries have sign changes, so only the first one should be reported
    signChange[12] = 4;
    signChange[13] = 5;
    signChange[14] = 6;

    // Sending data to GPU:
    dfloat *d_c_t, *d_c_test;
    int *d_signChange;
    cudaMalloc(&d_c_t, curve_length*sizeof(dfloat));
    cudaMalloc(&d_c_test, 10*sizeof(dfloat));
    cudaMalloc(&d_signChange, blocksPerWavelength*curve_length*sizeof(int));
    cudaMemcpy(d_c_t, c_t, curve_length*sizeof(dfloat), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_test, c_test, 10*sizeof(dfloat), cudaMemcpyHostToDevice);
    cudaMemcpy(d_signChange, signChange, blocksPerWavelength*curve_length*sizeof(int), cudaMemcpyHostToDevice);
    // Getting the entries where the sign first changes:
    kernel_first_sign_change<<<1, 1>>>(d_c_t, d_c_test, d_signChange, blocksPerWavelength, curve_length);
    cudaMemcpy(c_t, d_c_t, curve_length*sizeof(dfloat), cudaMemcpyDeviceToHost);

    printf("Testing results of sign evaluation. Should be 0, 2, 3, 4, 5:\n");
    for (int i=0; i<curve_length; ++i){
        printf("%f, ", c_t[i]);
    }
    printf("\n");
    return 0;
}

/* Tests that the determinant matrix is set up properly, and the correct sign changes are found.
    TODO: redo this for new determinant functions
*/
int testFullSignChange(){

    int curve_length = 5;
    int velocities_length = 300;
    const int blockSize = 256;
    int blocksPerWavelength = (velocities_length / blockSize)+1;

    dfloat *c_t = (dfloat*) calloc(curve_length, sizeof(dfloat));
    dfloat *c_test = (dfloat*) calloc(velocities_length, sizeof(dfloat));
    for (int i=0; i<velocities_length; ++i){
        c_test[i] = i;
    }

    int *neighbors, *d_signChange;
    cudaMalloc(&neighbors, blocksPerWavelength*curve_length*sizeof(int));
    cudaMalloc(&d_signChange, blocksPerWavelength*curve_length*sizeof(int));

    dfloat *d_c_t, *d_c_test;
    cudaMalloc(&d_c_t, curve_length*sizeof(dfloat));
    cudaMemcpy(d_c_t, c_t, curve_length*sizeof(dfloat), cudaMemcpyHostToDevice);
    cudaMalloc(&d_c_test, velocities_length*sizeof(dfloat));
    cudaMemcpy(d_c_test, c_test, velocities_length*sizeof(dfloat), cudaMemcpyHostToDevice);

    dim3 numBlocks(curve_length, blocksPerWavelength);
    /*
    cuDoubleComplex **d_matrices;
    cudaMalloc((void**)&d_matrices, curve_length*velocities_length*sizeof(cuDoubleComplex*));
    for (int i=0; i<curve_length*velocities_length; ++i){
        cudaMalloc((void**)&d_matrices[i], 3*sizeof(cuDoubleComplex));
    }
    
    MASWaves_cuda_theoretical_dispersion_curve<<<numBlocks, 256, 256>>>(neighbors, d_signChange, velocities_length);

    int *signChange = (int*) calloc(blocksPerWavelength*curve_length, sizeof(int));
    cudaMemcpy(signChange, d_signChange, blocksPerWavelength*curve_length*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i=0; i<blocksPerWavelength*curve_length; ++i){
        printf("%d, ", signChange[i]);
    }
    printf("\n");

    first_sign_change<<<1, 1>>>(d_c_t, d_c_test, d_signChange, blocksPerWavelength, curve_length);

    cudaMemcpy(c_t, d_c_t, curve_length*sizeof(dfloat), cudaMemcpyDeviceToHost);
    for (int i=0; i<curve_length; ++i){
        printf("%f, ", c_t[i]);
    }
    printf("\n");
    */
    return 0;

}

/* Tests the stiffness matrices are set up properly.
*/
int testDeterminantMatrixSetup(){

    int curve_length = 5;
    int velocities_length = 10;
    int n = 1;
    int size = 2*(n+1);
    const int blockSize = 256;
    int blocksPerWavelength = (velocities_length / blockSize)+1;

    dim3 numBlocks(curve_length, blocksPerWavelength);

    cuDoubleComplex **d_matrices;
    cuDoubleComplex *data, *d_data;
    int *d_infoArray, *d_pivotArray;

    //cuDoubleComplex **matrices;
    //matrices = (cuDoubleComplex**) malloc(curve_length*velocities_length*sizeof(cuDoubleComplex*));
    data = (cuDoubleComplex*) malloc(size*size*curve_length*velocities_length*sizeof(cuDoubleComplex));
    /*
    for (int i=0; i<curve_length*velocities_length; ++i){
        matrices[i] = (cuDoubleComplex*) ((char*) data+i*((size_t)3)*sizeof(cuDoubleComplex));
        for (int j=0; j<3; ++j){
            matrices[i][j] = make_cuDoubleComplex(i, 0);
        }
    }
    */

    cudaMalloc((void**)&d_matrices, curve_length*velocities_length*sizeof(cuDoubleComplex*));
    cudaMalloc(&d_data, size*size*curve_length*velocities_length*sizeof(cuDoubleComplex));
    cudaMalloc(&d_infoArray, curve_length*velocities_length*sizeof(int));
    cudaMalloc(&d_pivotArray, size*curve_length*velocities_length*sizeof(int));

    kernel_assign_matrices_to_data<<<1,1>>>(d_matrices, d_data, curve_length, velocities_length, n);
    kernel_matrix_fill_in_serial<<<1, 50>>>(velocities_length, curve_length, n, d_matrices);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    cublasZgetrfBatched(handle, size, d_matrices, size, d_pivotArray, d_infoArray, curve_length*velocities_length);

    //MASWaves_cuda_stiffness_matrix<<<numBlocks, blockSize>>>(velocities_length, curve_length, d_matrices);

    //cudaMemcpy(matrices, d_matrices, curve_length*velocities_length*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToHost);
    cudaMemcpy(data, d_data, size*size*curve_length*velocities_length*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    for (int i=0; i<curve_length*velocities_length; ++i){
        for (int j=0; j<size; ++j){
            for (int k=0; k<size; ++k){
                printf("%f, ", cuCreal(data[i*size*size + j*size + k]));
            }
            printf("\n");
        }
        printf("\n\n");
    }
    
    return 0;
}

/* Tests how the total end to end process performs with a long dispersion curve.
*/
int testScaling(int curve_size, dfloat wavelength){

    const int n = 6;
    dfloat h[n] = {1.0, 1.0, 2.0, 2.0, 4.0, 5.0};
    dfloat alpha[n+1] = {1440.0, 1440.0, 1440.0, 1440.0, 1440.0, 1440.0, 1440.0};
    dfloat beta[n+1] = {75.0, 90.0, 150.0, 180.0, 240.0, 290.0, 290.0};
    dfloat rho[n+1] = {1850.0, 1850.0, 1850.0, 1850.0, 1850.0, 1850.0, 1850.0};

    const int c_length = 1001;
    const int l_length = curve_size; //can be 42 for first 2 entries
    
    //60.0000, 35.3333 are first 2
    dfloat *lambda = (dfloat*) calloc(l_length, sizeof(dfloat));
    dfloat *c_curve = (dfloat*) calloc(l_length, sizeof(dfloat));
    for (int i=0; i<l_length; ++i){
        //lambda[i]   = 28.440;
        lambda[i]   = wavelength;
        c_curve[i]  = 0.1;
    }

    dfloat *c_array = (dfloat*) calloc(c_length, sizeof(dfloat));
    dfloat value = 0.5;
    
    for (int i=0; i<c_length; ++i){
        c_array[i] = value;
        value += 0.5;
    }

    curve_t *curve = (curve_t*) calloc(1, sizeof(curve_t));

    MASWA_setVariables(curve, c_array, h, alpha, beta, rho, n, c_length, l_length, c_curve, lambda);

    dfloat w = MASWA_inversion_CUDA(curve);
    
    for (int i=0; i<1; ++i){
        printf("%lf,", curve->c_t[i]);
    }
    printf("\n\n");
    
    return 0;
}

/*Runs test_scaling multiple times over increasing dispersion curves:
*/
int testScaling_full(){
    int result = 0;
    int trials = 10;
    double *time_spent = (double*) calloc(3*(trials+1), sizeof(double));
    clock_t begin, end;

    dfloat wavelengths[3] = {0.1, 28.440, 50.0};

    for (int j=0; j<3; ++j){
        begin = clock();
        result = testScaling(50, wavelengths[j]);
        end = clock();
        time_spent[j*(trials+1)] = (double)(end - begin) / CLOCKS_PER_SEC;
    }

    for (int i=1; i<trials+1; ++i){
        for (int j=0; j<3; ++j){
            begin = clock();
            result = testScaling((i)*50, wavelengths[j]);
            end = clock();
            time_spent[j*(trials+1) + i] = (double)(end - begin) / CLOCKS_PER_SEC;
        }
    }
    printf("[");
    for (int j=0; j<3; ++j){
        printf("%f, [", wavelengths[j]);
        for (int i=0; i<trials+1; ++i){
            printf("%f, ", time_spent[j*(trials+1) + i]);
        }
        printf("],");
    }
    printf("]\n\n");

    return 0;
}

/* Tests the misfit functions implemented in CUDA, both single block and multi block, for accuracy and performance.
*/
int testMisfitCuda(){

    int curve_length = 40;

    dfloat c_t[40] = {238.000000, 230.500000, 219.500000, 205.500000, 195.500000, 183.000000, 172.000000, 160.000000, 149.500000, 137.000000, 128.000000, 121.000000, 116.500000, 110.500000, 105.500000, 102.000000, 98.500000, 94.500000, 92.500000, 90.500000, 88.500000, 
                    87.000000, 86.000000, 84.500000, 83.500000, 82.500000, 82.000000, 81.000000, 80.000000, 79.500000, 78.500000, 78.000000, 78.000000, 77.500000, 77.000000, 76.500000, 76.000000, 76.000000, 75.500000, 75.500000};

    dfloat c_curve0[40] = {237.000000, 232.000000, 220.000000, 205.000000, 199.000000, 187.000000, 176.000000, 163.000000, 151.000000, 135.000000, 125.000000, 118.000000, 115.000000, 109.000000, 104.000000, 101.000000, 98.000000, 93.000000, 91.000000, 90.000000, 
                            88.000000, 87.000000, 86.000000, 85.000000, 84.000000, 83.000000, 82.000000, 81.000000, 79.000000, 78.000000, 77.000000, 77.000000, 77.000000, 76.000000, 76.000000, 75.000000, 74.000000, 73.000000, 73.000000, 73.000000};

    dfloat error = run_misfit(curve_length, c_t, c_curve0);

    return 0;

}
