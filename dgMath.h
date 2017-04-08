#ifndef DG_MATH_H__
#define DG_MATH_H__

#include <cmath>
#include <iostream>
#include <mkl_lapacke.h>

#include "array.h"

/** Returns the Chebyshev points of order p on [-1,1] */
darray chebyshev(int p);

/** Returns the Chebyshev points of order p in 3D on [-1,1]^3 */
darray chebyshev3D(int p);

/**
   Given an order p, computes the Gaussian quadrature points x 
   and weights w using the eigenvalues/eigenvectors of 
   Jacobi matrix from MATH 224A. They are on the domain [-1,1]
*/
int gaussQuad(int p, darray& x, darray& w);

/**
   Given an order p, computes the Gaussian quadrature points x3D 
   and weights w3D on the domain [-1,1]^3. Returns the total 
   number of points to evaluate
*/
int gaussQuad3D(int p, darray& x3D, darray& w3D);

#endif
