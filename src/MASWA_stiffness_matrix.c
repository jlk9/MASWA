#include "MASW.h"

/*The function MASWaves_stiffness_matrix assembles the system stiffness
  matrix of the stratified earth model that is used in the inversion
  analysis and computes its determinant.

  Note alpha, beta, and rho are all length n+1 while h is length n.
 */
dfloat MASWA_stiffness_matrix(dfloat c_test, dfloat k, dfloat *h, dfloat *alpha, dfloat *beta, dfloat *rho, int n){
    
    compfloat **K = (compfloat**) calloc(2*(n+1), sizeof(compfloat*));
    for (int i=0; i<2*(n+1); ++i){
        K[i] = (compfloat*) calloc(2*(n+1), sizeof(compfloat));
    }

    dfloat epsilon = 0.0001;
    while (tooClose(c_test, alpha, beta, n+1, epsilon) == 1){
        c_test *= 1-epsilon;
    }
    

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

    compfloat *Ke_halfspace = MASWA_Ke_halfspace(alpha[n], beta[n], rho[n], c_test, k);
    #define Ke_halfspace(r, c) (Ke_halfspace[(r)*2 + (c)])
    for (int r=0; r<2; ++r){
        for (int c=0; c<2; ++c){
            // Increment K with Ke_halfspace
            K[2*n + r][2*n + c] += Ke_halfspace(r, c);
        }
    }

    // real part of determinant
    dfloat D = real(hepta_determinant(K, 2*(n+1)));

    free(Ke_halfspace);
    for (int i=0; i<2*(n+1); ++i){
        free(K[i]);
    }
    free(K);

    return D;
}

/* Helper, used to check if c_test is too close to any entries in alpha or beta
   and thus requires modification.
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
                    matrix[i] = matrix[s];
                    matrix[s] = temp;
                    determinant *= -1.0;
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
