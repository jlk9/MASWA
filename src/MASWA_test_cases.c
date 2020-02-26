#include "MASW.h"

// Written by Joseph Kump (josek97@vt.edu). Last modified 01/31/2020.

/*
Suite of test cases for various functions in this package.
*/

/* Tests correctness of Ke_layer and Ke_halfspace. Not intended to evaluate
    performance or use with multiple processes.
*/
int testKeLayer(){
    const int n = 6;
    dfloat h[n] = {1.0, 1.0, 2.0, 2.0, 4.0, 5.0};
    dfloat alpha[n+1] = {1440.0, 1440.0, 1440.0, 1440.0, 1440.0, 1440.0, 1440.0};
    dfloat beta[n+1] = {75.0, 90.0, 150.0, 180.0, 240.0, 290.0, 290.0};
    dfloat rho[n+1] = {1850.0, 1850.0, 1850.0, 1850.0, 1850.0, 1850.0, 1850.0};
    dfloat c_test = 400.0;
    dfloat k = 0.1778;

    // Testing Ke_layer
    int w = 0;
    compfloat *Ke_layer = MASWA_Ke_layer(h[w], alpha[w], beta[w], rho[w], c_test, k);
    
    for (int i=0; i<16; i+=4){
        for (int j=0; j<4; ++j){
            printf("%lf%+lfi\t", real(Ke_layer[i+j]), imag(Ke_layer[i+j]));
        }
        printf("\n");
    }
    printf("\n");
    free(Ke_layer);

    // Testing Ke_halfspace
    compfloat *Ke_halfspace = MASWA_Ke_halfspace(alpha[n], beta[n], rho[n], c_test, k);
    
    for (int i=0; i<4; ++i){
        printf("%lf%+lfi\t", real(Ke_halfspace[i]), imag(Ke_halfspace[i]));
    }
    printf("\n\n");

    return 0;
}

/* Tests correctness of the stiffness matrix implementation. Not meant to evaluate
    performance or how function works with multiple processes.
*/
int testStiffnessMatrix(){

    const int n = 6;
    dfloat h[n] = {1.0, 1.0, 2.0, 2.0, 4.0, 5.0};
    dfloat alpha[n+1] = {1440.0, 1440.0, 1440.0, 1440.0, 1440.0, 1440.0, 1440.0};
    dfloat beta[n+1] = {75.0, 90.0, 150.0, 180.0, 240.0, 290.0, 290.0};
    dfloat rho[n+1] = {1850.0, 1850.0, 1850.0, 1850.0, 1850.0, 1850.0, 1850.0};
    dfloat c_test = 400.0;
    dfloat k = 0.1778;

    dfloat det = MASWA_stiffness_matrix(c_test, k, h, alpha, beta, rho, n);
    
    printf("\n%f\n", det);
    return 0;
}

/* Tests correctness of inversion on a realistic dataset with decreasing wavelengths
    and velocities on the dispersion curve.
*/
int testProcess(){

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    const int n = 6;
    dfloat h[n] = {1.0, 1.0, 2.0, 2.0, 4.0, 5.0};
    dfloat alpha[n+1] = {1440.0, 1440.0, 1440.0, 1440.0, 1440.0, 1440.0, 1440.0};
    dfloat beta[n+1] = {75.0, 90.0, 150.0, 180.0, 240.0, 290.0, 290.0};
    dfloat rho[n+1] = {1850.0, 1850.0, 1850.0, 1850.0, 1850.0, 1850.0, 1850.0};
    dfloat c_test = 400.0;

    const int c_length = 1001;
    const int l_length = 40;
    
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

    dfloat w = MASWA_inversion(curve);
    
    printf("The c_t results for rank %d are:\n", rank);
    for (int i=0; i<curve->curve_length; ++i){
        printf("%lf,", curve->c_t[i]);
    }

    printf("\n\nThe lambda results for rank %d are:\n", rank);
    for (int j=0; j<curve->curve_length; ++j){
        printf("%lf,", curve->lambda_t[j]);
    }
    
    printf("\n\nThe experimental results for rank %d are:\n", rank);
    for (int j=0; j<curve->curve_length; ++j){
        printf("%lf,", curve->c_curve0[j]);
    }
    printf("\n\n");

    
    if (rank == 0){
        printf("Misfit is %lf\n", w);
    }
    
    return 0;

}

/* Runs testProcess 11 times (accounting for potential compilation time in first run),
    and gathers the run times to gauge performance.
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
}

/* Tests correctness of inversion on a uniform dataset, to gauge strong and weak scaling
    capabilities of the algorithm with the load imbalancing of variable data.
*/
int testScaling(int curve_size, dfloat wavelength){

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    const int n = 6;
    dfloat h[n] = {1.0, 1.0, 2.0, 2.0, 4.0, 5.0};
    dfloat alpha[n+1] = {1440.0, 1440.0, 1440.0, 1440.0, 1440.0, 1440.0, 1440.0};
    dfloat beta[n+1] = {75.0, 90.0, 150.0, 180.0, 240.0, 290.0, 290.0};
    dfloat rho[n+1] = {1850.0, 1850.0, 1850.0, 1850.0, 1850.0, 1850.0, 1850.0};
    dfloat c_test = 400.0;
    dfloat k = 0.1778;

    const int c_length = 1001;
    const int l_length = curve_size; //can be 42 for first 2 entries
    
    dfloat *lambda = (dfloat*) calloc(l_length, sizeof(dfloat));
    dfloat *c_curve = (dfloat*) calloc(l_length, sizeof(dfloat));
    for (int i=0; i<l_length; ++i){
        lambda[i]   = wavelength;
        c_curve[i]  = 0.1;
    }

    dfloat *c_array = (dfloat*) calloc(c_length, sizeof(dfloat));
    dfloat value = 0.0;
    
    for (int i=0; i<c_length; ++i){
        c_array[i] = value;
        value += 0.5;
    }
    
    curve_t *curve = (curve_t*) calloc(1, sizeof(curve_t));
    
    MASWA_setVariables(curve, c_array, h, alpha, beta, rho, n, c_length, l_length, c_curve, lambda);

    dfloat w = MASWA_inversion(curve);

    for (int i=0; i<5; ++i){
        printf("%lf,", curve->c_t[i]);
    }
    printf("\n\n");

    return 0;
}

/*Runs test_scaling multiple times over increasing dispersion curves, used to assess weak scaling.
    Collects and prints runtimes to gauge performance.
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

