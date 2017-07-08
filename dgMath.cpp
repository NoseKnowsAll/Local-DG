#include "dgMath.h"

/** Performs the kronecker product C = kron(A,B) of two 2D matrices */
void kron(const darray& A, const darray& B, darray& C) {
  
  int ma = A.size(0);
  int na = A.size(1);
  int mb = B.size(0);
  int nb = B.size(1);
  
  kron(A, ma, na, B, mb, nb, C);
  
}

/** Performs the kronecker product C = kron(A,B) of two darrays */
void kron(const darray& A, int ma, int na,
	  const darray& B, int mb, int nb,
	  darray& C) {

  C.realloc(mb,ma, nb,na);

  for (int j1 = 0; j1 < na; ++j1) {
    for (int i1 = 0; i1 < ma; ++i1) {
      for (int j2 = 0; j2 < nb; ++j2) {
	for (int i2 = 0; i2 < mb; ++i2) {
	  C(i2,i1,j2,j1) = A(i1,j1)*B(i2,j2);
	}
      }
    }
  }
  
  // TODO: should I resize here also?
  //C.resize(mb*ma,nb*na);
}

/** Linearly interpolates input, cube dataset to get data value at a specified position */
double gridInterp(const darray& pos,
		  const darray& data, const darray& origins, const darray& deltas) {
  
  int dim = pos.size(0);
  double eps = 1e-8;
  
  darray index{dim};
  barray onInt{dim};
  int interpDims = dim;
  // Compute index into structured grid that position maps to
  for(int l = 0; l < dim; ++l) {
    index(l) = (pos(l) - origins(l))/deltas(l);
    if (index(l) < 0.0 || index(l) > data.size(l)-1) {
      std::cerr << "ERROR: Trying to interpolate to node that is outside of the dataset domain!" << std::endl;
      return -1.0;
    }
    if (std::abs(std::round(index(l))-index(l)) < eps*deltas(l)) {
      onInt(l) = true;
      interpDims--;
    }
    else
      onInt(l) = false;
  }
  
  // Index is perfectly aligned on grid already - no interpolation necessary
  if (interpDims == 0) {
    long long offset = 0;
    for (int l = 0; l < dim; ++l) {
      offset += data.stride(l)*(long long)std::round(index(l));
    }
    double toReturn = data[offset];
    return toReturn;
  }
  
  std::cout << interpDims << std::endl;
  
  // Compute weights = normalized volume of ND rectangular prism in direction of corner
  darray weights{1<<interpDims};
  for (int i = 0; i < (1<<interpDims); ++i) {
    weights(i) = 1.0;
  }
  
  int arrayDim = 0;
  for (int l = 0; l < dim; ++l) {
    if (!onInt(l)) {
      
      double factor0 = index(l) - std::floor(index(l));
      double factor1 = std::ceil(index(l)) - index(l);
      for (int i = 0; i < (1<<interpDims); ++i) {
	if ((i >> arrayDim) & 1) {
	  weights(i) *= factor0;
	}
	else {
	  weights(i) *= factor1;
	}
      }
      
      arrayDim++;
    }
    
  }

  std::cout << "weights = " << weights << std::endl;
  
  // Compute weighted sum of neighboring values
  double toReturn = 0.0;

  for (int i = 0; i < (1<<interpDims); ++i) {
    
    long long offset = 0;
    arrayDim = 0;
    for (int l = 0; l < dim; ++l) {
      if (onInt(l)) {
	offset += data.stride(l)*(long long)std::round(index(l));
      }
      else {
	long long localIndex;
	if ((i >> arrayDim) & 1)
	  localIndex = (long long)std::ceil(index(l));
	else
	  localIndex = (long long)std::floor(index(l));
	
	offset += data.stride(l)*localIndex;
	
	arrayDim++;
      }
    }

    // Actually compute weighted sum
    toReturn += weights(i)*data[offset];
  }
  return toReturn;
  
}

/** Edits cheby to contain the Chebyshev points of order p on [-1,1] */
int chebyshev1D(int p, darray& cheby) {
  
  cheby.realloc(1, p+1);
  for (int ix = 0; ix <= p; ++ix) {
    cheby(0, ix) = cos((p-ix)*M_PI/p);
  }
  return p+1;
  
}

/** Edits cheby2D to contain the Chebyshev points of order p in 2D on [-1,1]^2 */
int chebyshev2D(int p, darray& cheby2D) {
  
  darray cheby;
  chebyshev1D(p, cheby);
  
  cheby2D.realloc(2, p+1,p+1);
  for (int iy = 0; iy <= p; ++iy) {
    for (int ix = 0; ix <= p; ++ix) {
      cheby2D(0, ix,iy) = cheby(0,ix);
      cheby2D(1, ix,iy) = cheby(0,iy);
    }
  }
  
  return (p+1)*(p+1);
  
}

/** Edits cheby3D to contain the Chebyshev points of order p in 3D on [-1,1]^3 */
int chebyshev3D(int p, darray& cheby3D) {
  
  darray cheby;
  chebyshev1D(p, cheby);
  
  cheby3D.realloc(3, p+1,p+1,p+1);
  for (int iz = 0; iz <= p; ++iz) {
    for (int iy = 0; iy <= p; ++iy) {
      for (int ix = 0; ix <= p; ++ix) {
	cheby3D(0, ix,iy,iz) = cheby(0,ix);
	cheby3D(1, ix,iy,iz) = cheby(0,iy);
	cheby3D(2, ix,iy,iz) = cheby(0,iz);
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
int gaussQuad1D(int p, darray& x, darray& w) {
  int n = (int) ceil((p+1)/2.0);
  
  /* Initialize upper triangular (symmetric) Jacobi matrix */
  darray toEig{n,n};
  for (int i = 0; i < n-1; ++i) {
    double toInsert = (i+1)/sqrt(4*(i+1)*(i+1)-1);
    
    toEig(i, i+1) = toInsert;
    //toEig(i+1, i) = toInsert; // symmetric
  }
  
  /* Compute eigenvalue decomposition of Jacobi matrix */
  x.realloc(1,n);
  LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, toEig.data(), n, x.data());
  
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
  int n = gaussQuad1D(p, x1D, w1D);
  
  x2D.realloc(2, n,n);
  w2D.realloc(n,n);
  
  // Apply 1D quadrature to 2D cube
  for (int iy = 0; iy < n; ++iy) {
    for (int ix = 0; ix < n; ++ix) {
      x2D(0, ix,iy) = x1D(0,ix);
      x2D(1, ix,iy) = x1D(0,iy);
      
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
  int n = gaussQuad1D(p, x1D, w1D);
  
  x3D.realloc(3, n,n,n);
  w3D.realloc(n,n,n);
  
  // Apply 1D quadrature to 3D cube
  for (int iz = 0; iz < n; ++iz) {
    for (int iy = 0; iy < n; ++iy) {
      for (int ix = 0; ix < n; ++ix) {
	x3D(0, ix,iy,iz) = x1D(0,ix);
	x3D(1, ix,iy,iz) = x1D(0,iy);
	x3D(2, ix,iy,iz) = x1D(0,iz);
	
	w3D(ix,iy,iz) = w1D(ix)*w1D(iy)*w1D(iz);
      }
    }
  }
  
  return n*n*n;
}

/**
   Evaluates the 1D Legendre polynomials of order 0:p on [-1,1]
   at the points specified in x. Here, x is a 1 by nx vector
*/
void legendre(int p, const darray& x, darray& polys) {
  
  int nx = x.size(1);
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
    polys(i, 1) = x(0,i);
  }
  if (p == 1)
    return;
  
  // Three-term recurrence relationship
  for (int ip = 0; ip < p-1; ++ip) {
    for (int i = 0; i < nx; ++i) {
      polys(i,ip+2) = ((2*ip+3)*x(0,i)*polys(i,ip+1) - (ip+1)*polys(i,ip)) / (ip+2);
    }
  }
  return;
  
}

/**
   Evaluates the 1D derivative of Legendre polynomials of order 0:p 
   on [-1,1] at the points specified in x. Here, x is a 1 x nx vector.
   Assumes polys has already been initialized by legendre().
*/
void dlegendre(int p, const darray& x, const darray& polys, darray& dpolys) {
  
  int nx = x.size(1);
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
      dpolys(i,ip+2) = ((2*ip+3)*(x(0,i)*dpolys(i,ip+1) + polys(i,ip+1))
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
  darray x{1,nx};
  darray y{1,ny};
  for (int ix = 0; ix < nx; ++ix) {
    x(0,ix) = x2D(0, ix, 0);
  }
  for (int iy = 0; iy < ny; ++iy) {
    y(0,iy) = x2D(1, 0, iy);
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
  darray x{1,nx};
  darray y{1,ny};
  for (int ix = 0; ix < nx; ++ix) {
    x(0,ix) = x2D(0, ix, 0);
  }
  for (int iy = 0; iy < ny; ++iy) {
    y(0,iy) = x2D(1, 0, iy);
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
  darray x{1,nx};
  darray y{1,ny};
  darray z{1,nz};
  for (int ix = 0; ix < nx; ++ix) {
    x(0,ix) = x3D(0, ix, 0, 0);
  }
  for (int iy = 0; iy < ny; ++iy) {
    y(0,iy) = x3D(1, 0, iy, 0);
  }
  for (int iz = 0; iz < nz; ++iz) {
    z(0,iz) = x3D(2, 0, 0, iz);
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
  darray x{1,nx};
  darray y{1,ny};
  darray z{1,nz};
  for (int ix = 0; ix < nx; ++ix) {
    x(0,ix) = x3D(0, ix, 0, 0);
  }
  for (int iy = 0; iy < ny; ++iy) {
    y(0,iy) = x3D(1, 0, iy, 0);
  }
  for (int iz = 0; iz < nz; ++iz) {
    z(0,iz) = x3D(2, 0, 0, iz);
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
  lapack_int ipiv[nFrom];
  LAPACKE_dgesv(LAPACK_COL_MAJOR, nFrom, nFrom, 
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
  lapack_int ipiv[nFrom];
  LAPACKE_dgesv(LAPACK_COL_MAJOR, nFrom, nFrom, 
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
  lapack_int ipiv[nFrom];
  LAPACKE_dgesv(LAPACK_COL_MAJOR, nFrom, nFrom, 
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
