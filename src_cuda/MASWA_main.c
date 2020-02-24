#include "MASW.cuh"

int main(int argc, char**argv){

    
    
    double time_spent = 0.0;
    clock_t begin = clock();
    /*
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    printf("This is rank %d of %d. Getting started:\n", rank, size);
    */

    int result = testProcess();
    //dfloat thing = MASWaves_inversion_CUDA();
    
    //MPI_Finalize();
    
    clock_t end = clock();

	// calculate elapsed time by finding difference (end - begin) and
	// dividing the difference by CLOCKS_PER_SEC to convert to seconds
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;

	printf("(%d, %f), ", 1, time_spent);
    
    
    /*
    int length = 1010;

    dfloat *x, *d_x;
    x = (dfloat*) calloc(length, sizeof(dfloat));
    
    cudaMalloc(&d_x, length*sizeof(dfloat));
    cudaMemcpy(x, d_x, length*sizeof(dfloat), cudaMemcpyHostToDevice);
    run_kernel(length, d_x);

    for (int i=0; i<length; ++i){
        x[i] = 1.0;
    }

    cudaMemcpy(x, d_x, length*sizeof(dfloat), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    
    for (int i=0; i<length; ++i){
        printf("%lf  ", x[i]);
    }
    printf("\n");

    clock_t end = clock();

	// calculate elapsed time by finding difference (end - begin) and
	// dividing the difference by CLOCKS_PER_SEC to convert to seconds
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;

	printf("Took %f\n", time_spent);
    */
    return 0;
}
