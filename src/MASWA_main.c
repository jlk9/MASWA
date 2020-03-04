#include "MASW.h"

// Written by Joseph Kump (josek97@vt.edu). Last modified 01/27/2020

/* The main function called by the MASWA_mpi executable. Can be used to run MASWA_inversion,
    or one of the test functions available. 

    Currently no input arguments are used in this function.
    It returns 0 if run correctly.

*/
int main(int argc, char**argv){

    
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    testProcess();

    MPI_Finalize();
    
    return 0;
}
