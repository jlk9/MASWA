#include "MASW.cuh"

// Written by Joseph Kump (josek97@vt.edu). Last modified 01/27/2020

/* The main function called by the executable when it is compiled. Currently no arguments
    are implemented, so the tests or functions to use should be placed here. Returns the
    runtime for testing purposes.
*/
int main(int argc, char**argv){

    double time_spent = 0.0;
    clock_t begin = clock();
    
    int result = testProcess();
    
    clock_t end = clock();

	// calculate elapsed time by finding difference (end - begin) and
	// dividing the difference by CLOCKS_PER_SEC to convert to seconds
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;

	printf("(%d, %f), ", 1, time_spent);
    
    return 0;
}
