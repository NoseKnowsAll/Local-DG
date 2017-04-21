#ifndef DG_MATH_H__
#define DG_MATH_H__

#include <cmath>
#include <iostream>
#include <mkl_lapacke.h>
#include <mkl.h>

#include "array.h"

/** Edits cheby to contain the Chebyshev points of order p on [-1,1] */
int chebyshev(int p, darray& cheby);

/** Edits cheby2D to contain the Chebyshev points of order p in 2D on [-1,1]^2 */
int chebyshev2D(int p, darray& cheby2D);

/** Edits cheby3D to contain the Chebyshev points of order p in 3D on [-1,1]^3 */
int chebyshev3D(int p, darray& cheby3D);

/**
   Given an order p, computes the Gaussian quadrature points x 
   and weights w using the eigenvalues/eigenvectors of 
   Jacobi matrix from MATH 224A. They are on the domain [-1,1]
*/
int gaussQuad(int p, darray& x, darray& w);

/**
   Given an order p, computes the Gaussian quadrature points x2D 
   and weights w2D on the domain [-1,1]^2. Returns the total
   number of points to evaluate. 
*/
int gaussQuad2D(int p, darray& x2D, darray& w2D);

/**
   Given an order p, computes the Gaussian quadrature points x3D 
   and weights w3D on the domain [-1,1]^3. Returns the total 
   number of points to evaluate
*/
int gaussQuad3D(int p, darray& x3D, darray& w3D);

/**
   Evaluates the 1D Legendre polynomials of order 0:p on [-1,1]
   at the points specified in x
*/
void legendre(int p, const darray& x, darray& polys);

/**
   Evaluates the 1D derivative of Legendre polynomials of order 0:p 
   on [-1,1] at the points specified in x.
   Assumes polys has already been initialized by legendre().
*/
void dlegendre(int p, const darray& x, const darray& polys, darray& dpolys);

/**
   Evaluates the 2D Legendre polynomials of order 0:p in each dimension on the 
   reference element [-1,1]^2 at the points specified in x2D.
*/
void legendre2D(int p, const darray& x2D, darray& l2D);

/**
   Evaluates the 2D derivatives of Legendre polynomials of order 0:p in each 
   dimension on the reference element [-1,1]^2 at the points specified in x2D
*/
void dlegendre2D(int p, const darray& x2D, darray& dl2D);

/**
   Evaluates the 3D Legendre polynomials of order 0:p in each dimension on the 
   reference element [-1,1]^3 at the points specified in x3D
*/
void legendre3D(int p, const darray& x3D, darray& l3D);

/**
   Evaluates the 3D derivatives of Legendre polynomials of order 0:p in each 
   dimension on the reference element [-1,1]^3 at the points specified in x3D
*/
void dlegendre3D(int p, const darray& x3D, darray& dl3D);

/**
   Computes an interpolation matrix from a set of 1D points to another set of 1D points.
   Assumes that points interpolating from provide enough accuracy (aka - 
   they are well spaced out and of high enough order), and define an interval.
   Points interpolating onto can be of any size, but must be defined on that same interval.
*/
void interpolationMatrix1D(const darray& xFrom, const darray& xTo, darray& INTERP);

/**
   Computes an interpolation matrix from a set of 2D points to another set of 2D points.
   Assumes that points interpolating from provide enough accuracy (aka - 
   they are well spaced out and of high enough order), and define a square.
   Points interpolating onto can be of any size, but must be defined on that same square.
*/
void interpolationMatrix2D(const darray& xFrom, const darray& xTo, darray& INTERP);

/**
   Computes an interpolation matrix from a set of 3D points to another set of 3D points.
   Assumes that points interpolating from provide enough accuracy (aka - 
   they are well spaced out and of high enough order), and define a cube.
   Points interpolating onto can be of any size, but must be defined on that same cube.
*/
void interpolationMatrix3D(const darray& xFrom, const darray& xTo, darray& INTERP);

#endif
