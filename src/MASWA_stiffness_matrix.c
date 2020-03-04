#include "MASW.h"

// Written by Joseph Kump (josek97@vt.edu). Last modified 12/15/2019.

/*The function MASWaves_stiffness_matrix assembles the system stiffness
  matrix of the stratified earth model that is used in the inversion
  analysis and computes its determinant.

  Note alpha, beta, and rho are all length n+1 while h is length n
  since we assume infinite half space as bottom layer.  

  Inputs:
  c_test    the array of test velocities from a curve object
  k         the wavelength currently being evaluated by MASW, inverted
  h         the array of thicknesses of stiffness layers
  alpha     the array of compressional wave velocities
  beta      the array of shear wave velocities
  rho       the array of layer densities
  n         the number of finite thickness layers in the ground model   

  Output: 
  D     the real component of the stiffness matrix determinant, used in the algorithm
            to find the theoretical velocity of MASW

 */
dfloat MASWA_stiffness_matrix(dfloat c_test, dfloat k, dfloat *h, dfloat *alpha, dfloat *beta, dfloat *rho, int n){
    
    // First we allocate the stiffness matrix row by row:
    compfloat **K = (compfloat**) calloc(2*(n+1), sizeof(compfloat*));
    for (int i=0; i<2*(n+1); ++i){
        K[i] = (compfloat*) calloc(2*(n+1), sizeof(compfloat));
    }

    // Then we check that the test velocity is not too close to the model velocities:
    dfloat epsilon = 0.0001;
    while (tooClose(c_test, alpha, beta, n+1, epsilon) == 1){
        c_test *= 1-epsilon; // shrink c_test so it's different from model velocity
    }
    
    // Now we add individual Ke layers to the stiffness matrix, going through each finite
    // thickness layer (4x4 blocks per layer):
    for (int j=0; j<n; ++j){

        compfloat *Ke = MASWA_Ke_layer(h[j], alpha[j], beta[j], rho[j], c_test, k);
        #define Ke(r, c) (Ke[(r)*4 + (c)])
        
        for (int r=0; r<4; ++r){
            for (int c=0; c<4; ++c){
                // Increment K with Ke
                K[2*j + r][2*j + c] += Ke(r, c);
            }
        }
        free(Ke);
    }

    // Then we add the halfspace representing the last infinite thickness layer (2x2 block for halfspace):
    compfloat *Ke_halfspace = MASWA_Ke_halfspace(alpha[n], beta[n], rho[n], c_test, k);
    #define Ke_halfspace(r, c) (Ke_halfspace[(r)*2 + (c)])
    for (int r=0; r<2; ++r){
        for (int c=0; c<2; ++c){
            // Increment K with Ke_halfspace
            K[2*n + r][2*n + c] += Ke_halfspace(r, c);
        }
    }

    // Finally compute the real part of the determinant:
    dfloat D = real(hepta_determinant(K, 2*(n+1)));

    // Cleanup of memory
    free(Ke_halfspace);
    for (int i=0; i<2*(n+1); ++i){
        free(K[i]);
    }
    free(K);

    return D; // return the real part of the determinant
}

/* Helper, used to check if c_test is too close to any entries in alpha or beta
   and thus requires modification.
   
   Inputs:
   c_test   the test velocity
   alpha    the compressional wave velocities
   beta     the shear wave velocities
   length   the length of alpha and beta
   epsilon  the error tolerance required (if c_test is within epsilon of any model
                velocity then it is too close)
                
    Output:
        1 if the test velocity is too close to any of the velocity model parameters, 0 if
            it is not too close
*/
int tooClose(dfloat c_test, dfloat *alpha, dfloat *beta, int length, dfloat epsilon){
    for (int i=0; i<length; ++i){
        if (abs(c_test-alpha[i]) < epsilon || abs(c_test-beta[i]) < epsilon){
            return 1;
        }
    }
    return 0;
}

/* Helper, used to get the determinant via Guassian row reduction.
    This one takes advantage of the stiffness matrix being heptadiagonal,
    which brings it from O(N^3) to O(N).
    
    Inputs:
    matrix  the 2D array storing the entries of a stiffness matrix. Assumed to be stored
                row-first, and to have a banded heptadiagonal structure
    size    the dimension of the matrix's rows and columns
    
    Output:
    determinant     the determinant of this complex, banded, heptadiagonal matrix
 */
compfloat hepta_determinant(compfloat **matrix, int size){

    compfloat determinant = 1.0;
    
    for (int i=0; i<size; ++i){

        // Due to the heptadiagonal shape, we do not have to consider rows and
        // columns over three indices beyond current diagonal entry - they will
        // always be 0.
        int end = i+4;
        if (end > size){
            end = size;
        }

        // Row switching if we have a 0 entry on the diagonal
        if (matrix[i][i] == 0.0){
            for (int s=i+1; s<end; ++s){
                if (matrix[s][i] != 0.0){
                    compfloat *temp = matrix[i];
                    matrix[i]       = matrix[s];
                    matrix[s]       = temp;
                    determinant    *= -1.0;
                    break;
                }
            }
            // If we reach the end of the for loop and it's still 0, then the matrix is singular
            if (matrix[i][i] == 0.0){
                return 0.0;
            }
        }

        // Gaussian elimination for rows below this one:
        for (int j=i+1; j<end; ++j){
            compfloat coeff = matrix[j][i] / matrix[i][i];
            for (int k=i+1; k<end; ++k){
                matrix[j][k] = matrix[j][k] - coeff * matrix[i][k];
            }
        }

        // Modify determinant
        determinant *= matrix[i][i];
    }

    return determinant;
}

