#include "dgMath.h"

/** Returns the Chebyshev points of order p on [-1,1] */
darray chebyshev(int p) {
  
  darray cheby{p+1};
  for (int i = 0; i <= p; ++i) {
    cheby(i) = cos((p-i)*M_PI/p);
  }
  return cheby;
  
}

/** Returns the Chebyshev points of order p in 3D on [-1,1]^3 */
darray chebyshev3D(int p) {
  
  darray cheby = chebyshev(p);
  
  //int nNodes = (p+1)*(p+1)*(p+1);
  darray cheby3D{3, p+1,p+1,p+1};
  for (int iz = 0; iz <= p; ++iz) {
    for (int iy = 0; iy <= p; ++iy) {
      for (int ix = 0; ix <= p; ++ix) {
	//int vID = ix+iy*(p+1)+iz*(p+1)*(p+1);
	
	cheby3D(0, ix,iy,iz) = cheby(ix);
	cheby3D(1, ix,iy,iz) = cheby(iy);
	cheby3D(2, ix,iy,iz) = cheby(iz);
      }
    }
  }
  
  // TODO: should we reshape into 1D?
  //cheby3D.resize(3, nNodes);
  return cheby3D;
  
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

/**
   Given an order p, computes the Gaussian quadrature points x3D 
   and weights w3D on the domain [-1,1]^3. Returns the total
   number of points to evaluate. 
*/
int gaussQuad3D(int p, darray& x3D, darray& w3D) {
  
  // Get 1D quadrature pts/weights
  darray x1D, w1D;
  int n = gaussQuad(p, x1D, w1D);
  
  x3D.realloc(3, n,n,n);
  w3D.realloc(n,n,n);
  
  // Apply 1D quadrature to 3D cube
  for (int iz = 0; iz < n; ++iz) {
    for (int iy = 0; iy < n; ++iy) {
      for (int ix = 0; ix < n; ++ix) {
	x3D(0, ix,iy,iz) = x1D(ix);
	x3D(1, ix,iy,iz) = x1D(iy);
	x3D(2, ix,iy,iz) = x1D(iz);
	
	w3D(ix,iy,iz) = w1D(ix)*w1D(iy)*w1D(iz);
      }
    }
  }
  
  int totalPts = n*n*n;
  // TODO: should we reshape to 1D?
  //x3D.reshape(3, totalPts);
  //w3D.reshape(totalPts);
  return totalPts;
}
