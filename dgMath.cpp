#include "dgMath.h"

/** Returns the Chebyshev points of order p on [-1,1] */
darray chebyshev(int p) {
  
  darray cheby{p+1};
  for (int i = 0; i <= p; ++i) {
    cheby(i) = cos((p-i)*M_PI/p);
  }
  return cheby;
  
}

/**
   Given an order p, computes the Gaussian quadrature points x 
   and weights w using the eigenvalues/eigenvectors of 
   Jacobi matrix from MATH 224A. They are on the domain [-1,1]
*/
int gaussQuad(int p, darray& x, darray& w) {
  int n = (int) ceil((p+1)/2.0);
  
  /* Initialize upper triangular (symmetric) Jacobi matrix */
  darray toEig{n,n};
  for (int i = 0; i < n-1; ++i) {
    double toInsert = (i+1)/sqrt(4*(i+1)*(i+1)-1);
    
    toEig(i, i+1) = toInsert;
    //toEig(i+1, i) = toInsert; // symmetric
  }
  
  /* Compute eigenvalue decomposition of Jacobi matrix */
  x.realloc(n);
  int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, toEig.data(), n, x.data());
  
  /* Compute quadrature weights from eigenvectors */
  w.realloc(n);
  for (int i = 0; i < n; ++i) {
    w(i) = 2*toEig(0,i)*toEig(0,i);
  }
  
  return n;
}
