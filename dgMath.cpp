#include "dgMath.h"

/** Edits cheby to contain the Chebyshev points of order p on [-1,1] */
int chebyshev(int p, darray& cheby) {
  
  cheby.realloc(p+1);
  for (int i = 0; i <= p; ++i) {
    cheby(i) = cos((p-i)*M_PI/p);
  }
  return p+1;
  
}

/** Edits cheby2D to contain the Chebyshev points of order p in 2D on [-1,1]^2 */
int chebyshev2D(int p, darray& cheby2D) {
  
  darray cheby;
  chebyshev(p, cheby);
  
  cheby2D.realloc(2, p+1,p+1);
  for (int iy = 0; iy <= p; ++iy) {
    for (int ix = 0; ix <= p; ++ix) {
      cheby2D(0, ix,iy) = cheby(ix);
      cheby2D(1, ix,iy) = cheby(iy);
    }
  }
  
  return (p+1)*(p+1);
  
}

/** Edits cheby3D to contain the Chebyshev points of order p in 3D on [-1,1]^3 */
int chebyshev3D(int p, darray& cheby3D) {
  
  darray cheby;
  chebyshev(p, cheby);
  
  cheby3D.realloc(3, p+1,p+1,p+1);
  for (int iz = 0; iz <= p; ++iz) {
    for (int iy = 0; iy <= p; ++iy) {
      for (int ix = 0; ix <= p; ++ix) {
	cheby3D(0, ix,iy,iz) = cheby(ix);
	cheby3D(1, ix,iy,iz) = cheby(iy);
	cheby3D(2, ix,iy,iz) = cheby(iz);
      }
    }
  }
  
  return (p+1)*(p+1)*(p+1);
  
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
   Given an order p, computes the Gaussian quadrature points x2D 
   and weights w2D on the domain [-1,1]^2. Returns the total
   number of points to evaluate. 
*/
int gaussQuad2D(int p, darray& x2D, darray& w2D) {
  
  // Get 1D quadrature pts/weights
  darray x1D, w1D;
  int n = gaussQuad(p, x1D, w1D);
  
  x2D.realloc(2, n,n);
  w2D.realloc(n,n);
  
  // Apply 1D quadrature to 2D cube
  for (int iy = 0; iy < n; ++iy) {
    for (int ix = 0; ix < n; ++ix) {
      x2D(0, ix,iy) = x1D(ix);
      x2D(1, ix,iy) = x1D(iy);
      
      w2D(ix,iy) = w1D(ix)*w1D(iy);
    }
  }
  
  return n*n;
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
  
  return n*n*n;
}

/**
   Evaluates the 1D Legendre polynomials of order 0:p on [-1,1]
   at the points specified in x.
*/
void legendre(int p, const darray& x, darray& polys) {
  
  int nx = x.size(0);
  if (p < 0) {
    std::cerr << "ERROR: legendre polynomial order must be nonnegative!" << std::endl;
    polys.realloc(nx, 1);
    return;
  }
  
  polys.realloc(nx, p+1);
  
  // Initialize p_0(x) = 1
  for (int i = 0; i < nx; ++i) {
    polys(i, 0) = 1.0;
  }
  if (p == 0)
    return;
  // Initialize p_1(x) = x
  for (int i = 0; i < nx; ++i) {
    polys(i, 1) = x(i);
  }
  if (p == 1)
    return;
  
  // Three-term recurrence relationship
  for (int ip = 0; ip < p-1; ++ip) {
    for (int i = 0; i < nx; ++i) {
      polys(i,ip+2) = ((2*ip+3)*x(i)*polys(i,ip+1) - (ip+1)*polys(i,ip)) / (ip+2);
    }
  }
  return;
  
}

/**
   Evaluates the 1D derivative of Legendre polynomials of order 0:p 
   on [-1,1] at the points specified in x.
   Assumes polys has already been initialized by legendre().
*/
void dlegendre(int p, const darray& x, const darray& polys, darray& dpolys) {
  
  int nx = x.size(0);
  if (p < 0) {
    std::cerr << "ERROR: legendre polynomial order must be nonnegative!" << std::endl;
    dpolys.realloc(nx, 1);
    return;
  }
  
  dpolys.realloc(nx, p+1);
  
  // Initialize dp_0(x) = 0
  for (int i = 0; i < nx; ++i) {
    dpolys(i, 0) = 0.0;
  }
  if (p == 0)
    return;
  // Initialize dp_1(x) = 1
  for (int i = 0; i < nx; ++i) {
    dpolys(i, 1) = 1.0;
  }
  if (p == 1)
    return;
  
  // Three-term recurrence relationship
  for (int ip = 0; ip < p-1; ++ip) {
    for (int i = 0; i < nx; ++i) {
      dpolys(i,ip+2) = ((2*ip+3)*(x(i)*dpolys(i,ip+1) + polys(i,ip+1))
			- (ip+1)*dpolys(i,ip)) / (ip+2);
    }
  }
  
}

/**
   Evaluates the 2D Legendre polynomials of order 0:p in each dimension on the 
   reference element [-1,1]^2 at the points specified in x2D
*/
void legendre2D(int p, const darray& x2D, darray& l2D) {
  
  int nx = x2D.size(1);
  int ny = x2D.size(2);
  if (p < 0) {
    std::cerr << "ERROR: legendre polynomial order must be nonnegative!" << std::endl;
    l2D.realloc(nx*ny, 1,1);
    return;
  }
  
  // Initialize 1D arrays from 2D input
  darray x{nx};
  darray y{ny};
  for (int ix = 0; ix < nx; ++ix) {
    x(ix) = x2D(0, ix, 0);
  }
  for (int iy = 0; iy < ny; ++iy) {
    y(iy) = x2D(1, 0, iy);
  }
  
  // Compute 1D Legendre polynomials
  darray lx,ly;
  legendre(p, x, lx);
  legendre(p, y, ly);
  
  // Combine 1D Legendre polynomials into 2D polynomial
  l2D.realloc(nx, ny, p+1, p+1);
  
  for (int ipy = 0; ipy <= p; ++ipy) {
    for (int ipx = 0; ipx <= p; ++ipx) {
      for (int iy = 0; iy < ny; ++iy) {
	for (int ix = 0; ix < nx; ++ix) {
	  l2D(ix,iy,ipx,ipy) = lx(ix,ipx)*ly(iy,ipy);
	}
      }
    }
  }
  
  l2D.resize(nx*ny, p+1,p+1);
  
}

/**
   Evaluates the 2D derivatives of Legendre polynomials of order 0:p in each 
   dimension on the reference element [-1,1]^2 at the points specified in x2D.
*/
void dlegendre2D(int p, const darray& x2D, darray& dl2D) {
  
  int nx = x2D.size(1);
  int ny = x2D.size(2);
  if (p < 0) {
    std::cerr << "ERROR: legendre polynomial order must be nonnegative!" << std::endl;
    dl2D.realloc(nx*ny, 1, 1, 2);
    return;
  }
  
  // Initialize 1D arrays from 2D input
  darray x{nx};
  darray y{ny};
  for (int ix = 0; ix < nx; ++ix) {
    x(ix) = x2D(0, ix, 0);
  }
  for (int iy = 0; iy < ny; ++iy) {
    y(iy) = x2D(1, 0, iy);
  }
  
  // Compute 1D Legendre polynomials
  darray lx, ly;
  legendre(p, x, lx);
  legendre(p, y, ly);
  // Compute 1D derivatives of Legendre polynomials
  darray dlx, dly;
  dlegendre(p, x, lx, dlx);
  dlegendre(p, y, ly, dly);
  
  // Combine 1D Legendre polynomials into 2D polynomial
  dl2D.realloc(nx, ny, p+1, p+1, 2);
  
  // x
  for (int ipy = 0; ipy <= p; ++ipy) {
    for (int ipx = 0; ipx <= p; ++ipx) {
      for (int iy = 0; iy < ny; ++iy) {
	for (int ix = 0; ix < nx; ++ix) {
	  dl2D(ix,iy,ipx,ipy,0) = dlx(ix,ipx)*ly(iy,ipy);
	}
      }
    }
  }
  // y
  for (int ipy = 0; ipy <= p; ++ipy) {
    for (int ipx = 0; ipx <= p; ++ipx) {
      for (int iy = 0; iy < ny; ++iy) {
	for (int ix = 0; ix < nx; ++ix) {
	  dl2D(ix,iy,ipx,ipy,1) = lx(ix,ipx)*dly(iy,ipy);
	}
      }
    }
  }
  
  dl2D.resize(nx*ny, p+1,p+1, 2);
  
}

/**
   Evaluates the 3D Legendre polynomials of order 0:p in each dimension on the 
   reference element [-1,1]^3 at the points specified in x3D
*/
void legendre3D(int p, const darray& x3D, darray& l3D) {
  
  int nx = x3D.size(1);
  int ny = x3D.size(2);
  int nz = x3D.size(3);
  if (p < 0) {
    std::cerr << "ERROR: legendre polynomial order must be nonnegative!" << std::endl;
    l3D.realloc(nx*ny*nz, 1,1,1);
    return;
  }
  
  // Initialize 1D arrays from 3D input
  darray x{nx};
  darray y{ny};
  darray z{nz};
  for (int ix = 0; ix < nx; ++ix) {
    x(ix) = x3D(0, ix, 0, 0);
  }
  for (int iy = 0; iy < ny; ++iy) {
    y(iy) = x3D(1, 0, iy, 0);
  }
  for (int iz = 0; iz < nz; ++iz) {
    z(iz) = x3D(2, 0, 0, iz);
  }
  
  // Compute 1D Legendre polynomials
  darray lx,ly,lz;
  legendre(p, x, lx);
  legendre(p, y, ly);
  legendre(p, z, lz);
  
  // Combine 1D Legendre polynomials into 3D polynomial
  l3D.realloc(nx, ny, nz, p+1, p+1, p+1);
  
  for (int ipz = 0; ipz <= p; ++ipz) {
    for (int ipy = 0; ipy <= p; ++ipy) {
      for (int ipx = 0; ipx <= p; ++ipx) {
	for (int iz = 0; iz < nz; ++iz) {
	  for (int iy = 0; iy < ny; ++iy) {
	    for (int ix = 0; ix < nx; ++ix) {
	      l3D(ix,iy,iz,ipx,ipy,ipz) = lx(ix,ipx)*ly(iy,ipy)*lz(iz,ipz);
	    }
	  }
	}
      }
    }
  }
  
  l3D.resize(nx*ny*nz, p+1,p+1,p+1);
  
}

/**
   Evaluates the 3D derivatives of Legendre polynomials of order 0:p in each 
   dimension on the reference element [-1,1]^3 at the points specified in x3D
*/
void dlegendre3D(int p, const darray& x3D, darray& dl3D) {
  
  int nx = x3D.size(1);
  int ny = x3D.size(2);
  int nz = x3D.size(3);
  if (p < 0) {
    std::cerr << "ERROR: legendre polynomial order must be nonnegative!" << std::endl;
    dl3D.realloc(nx*ny*nz,1,1,1,3);
    return;
  }
  
  // Initialize 1D arrays from 3D input
  darray x{nx};
  darray y{ny};
  darray z{nz};
  for (int ix = 0; ix < nx; ++ix) {
    x(ix) = x3D(0, ix, 0, 0);
  }
  for (int iy = 0; iy < ny; ++iy) {
    y(iy) = x3D(1, 0, iy, 0);
  }
  for (int iz = 0; iz < nz; ++iz) {
    z(iz) = x3D(2, 0, 0, iz);
  }
  
  // Compute 1D Legendre polynomials
  darray lx,ly,lz;
  legendre(p, x, lx);
  legendre(p, y, ly);
  legendre(p, z, lz);
  // Compute 1D derivatives of Legendre polynomials
  darray dlx,dly,dlz;
  dlegendre(p, x, lx, dlx);
  dlegendre(p, y, ly, dly);
  dlegendre(p, z, lz, dlz);
  
  // Combine 1D Legendre polynomials into 3D polynomial
  dl3D.realloc(nx, ny, nz, p+1, p+1, p+1, 3);
  
  // x
  for (int ipz = 0; ipz <= p; ++ipz) {
    for (int ipy = 0; ipy <= p; ++ipy) {
      for (int ipx = 0; ipx <= p; ++ipx) {
	for (int iz = 0; iz < nz; ++iz) {
	  for (int iy = 0; iy < ny; ++iy) {
	    for (int ix = 0; ix < nx; ++ix) {
	      dl3D(ix,iy,iz,ipx,ipy,ipz,0) = dlx(ix,ipx)*ly(iy,ipy)*lz(iz,ipz);
	    }
	  }
	}
      }
    }
  }
  // y
  for (int ipz = 0; ipz <= p; ++ipz) {
    for (int ipy = 0; ipy <= p; ++ipy) {
      for (int ipx = 0; ipx <= p; ++ipx) {
	for (int iz = 0; iz < nz; ++iz) {
	  for (int iy = 0; iy < ny; ++iy) {
	    for (int ix = 0; ix < nx; ++ix) {
	      dl3D(ix,iy,iz,ipx,ipy,ipz,1) = lx(ix,ipx)*dly(iy,ipy)*lz(iz,ipz);
	    }
	  }
	}
      }
    }
  }
  // z
  for (int ipz = 0; ipz <= p; ++ipz) {
    for (int ipy = 0; ipy <= p; ++ipy) {
      for (int ipx = 0; ipx <= p; ++ipx) {
	for (int iz = 0; iz < nz; ++iz) {
	  for (int iy = 0; iy < ny; ++iy) {
	    for (int ix = 0; ix < nx; ++ix) {
	      dl3D(ix,iy,iz,ipx,ipy,ipz,2) = lx(ix,ipx)*ly(iy,ipy)*dlz(iz,ipz);
	    }
	  }
	}
      }
    }
  }
  
  dl3D.resize(nx*ny*nz, p+1,p+1,p+1, 3);
  
}

/**
   Computes an interpolation matrix from a set of 1D points to another set of 1D points.
   Assumes that points interpolating from provide enough accuracy (aka - 
   they are well spaced out and of high enough order), and define an interval.
   Points interpolating onto can be of any size, but must be defined on that same interval.
*/
void interpolationMatrix1D(const darray& xFrom, const darray& xTo, darray& INTERP) {
  
  // Create nodal representation of the reference bases
  int order = xFrom.size(1) - 1; // Assumes order = (size of xFrom)-1
  
  int nFrom = xFrom.size(1);
  darray lFrom;
  legendre(order, xFrom, lFrom);
  
  darray coeffsPhi{order+1,order+1};
  for (int ipx = 0; ipx <= order; ++ipx) {
    coeffsPhi(ipx,ipx) = 1.0;
  }
  MKL_INT ipiv[nFrom];
  int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, nFrom, nFrom, 
			   lFrom.data(), nFrom, ipiv, coeffsPhi.data(), nFrom);
  
  // Compute reference bases on the output points
  int nTo = xTo.size(1);
  darray lTo;
  legendre(order, xTo, lTo);
  
  // Construct interpolation matrix = lTo*coeffsPhi
  INTERP.realloc(nTo, nFrom);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      nTo, nFrom, nFrom, 1.0, lTo.data(), nTo, 
	      coeffsPhi.data(), nFrom, 0.0, INTERP.data(), nTo);
  
}

/**
   Computes an interpolation matrix from a set of 2D points to another set of 2D points.
   Assumes that points interpolating from provide enough accuracy (aka - 
   they are well spaced out and of high enough order), and define a square.
   Points interpolating onto can be of any size, but must be defined on that same square.
*/
void interpolationMatrix2D(const darray& xFrom, const darray& xTo, darray& INTERP) {
  
  // Create nodal representation of the reference bases
  int order = xFrom.size(1) - 1; // Assumes order = (size of each dimension of xFrom)-1
  
  int nFrom = xFrom.size(1)*xFrom.size(2);
  darray lFrom;
  legendre2D(order, xFrom, lFrom);
  
  darray coeffsPhi{order+1,order+1,order+1,order+1};
  for (int ipy = 0; ipy <= order; ++ipy) {
    for (int ipx = 0; ipx <= order; ++ipx) {
      coeffsPhi(ipx,ipy,ipx,ipy) = 1.0;
    }
  }
  MKL_INT ipiv[nFrom];
  int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, nFrom, nFrom, 
			   lFrom.data(), nFrom, ipiv, coeffsPhi.data(), nFrom);
  
  // Compute reference bases on the output points
  int nTo = xTo.size(1)*xTo.size(2);
  darray lTo;
  legendre2D(order, xTo, lTo);
  
  // Construct interpolation matrix = lTo*coeffsPhi
  INTERP.realloc(nTo, nFrom);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      nTo, nFrom, nFrom, 1.0, lTo.data(), nTo, 
	      coeffsPhi.data(), nFrom, 0.0, INTERP.data(), nTo);
  
}

/**
   Computes an interpolation matrix from a set of 3D points to another set of 3D points.
   Assumes that points interpolating from provide enough accuracy (aka - 
   they are well spaced out and of high enough order), and define a cube.
   Points interpolating onto can be of any size, but must be defined on that same cube.
*/
void interpolationMatrix3D(const darray& xFrom, const darray& xTo, darray& INTERP) {
  
  // Create nodal representation of the reference bases
  int order = xFrom.size(1) - 1; // Assumes order = (size of each dimension of xFrom)-1
  
  int nFrom = xFrom.size(1)*xFrom.size(2)*xFrom.size(3);
  darray lFrom;
  legendre3D(order, xFrom, lFrom);
  
  darray coeffsPhi{order+1,order+1,order+1,order+1,order+1,order+1};
  for (int ipz = 0; ipz <= order; ++ipz) {
    for (int ipy = 0; ipy <= order; ++ipy) {
      for (int ipx = 0; ipx <= order; ++ipx) {
	coeffsPhi(ipx,ipy,ipz,ipx,ipy,ipz) = 1.0;
      }
    }
  }
  MKL_INT ipiv[nFrom];
  int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, nFrom, nFrom, 
			   lFrom.data(), nFrom, ipiv, coeffsPhi.data(), nFrom);
  
  // Compute reference bases on the output points
  int nTo = xTo.size(1)*xTo.size(2)*xTo.size(3);
  darray lTo;
  legendre3D(order, xTo, lTo);
  
  // Construct interpolation matrix = lTo*coeffsPhi
  INTERP.realloc(nTo, nFrom);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      nTo, nFrom, nFrom, 1.0, lTo.data(), nTo, 
	      coeffsPhi.data(), nFrom, 0.0, INTERP.data(), nTo);
  
}
