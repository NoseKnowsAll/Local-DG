#include <iostream>
#include <mkl_lapacke.h>
#include <cmath>

#include "array.h"
#include "dgMath.h"
#include "mesh.h"
#include "solver.h"

/** Main driver function */
int main(int argc, char *argv[]) {
  
  Mesh mesh{};
  std::cout << mesh << std::endl;
  
  int p = 4;
  double tf = 2.0;
  Solver dgSolver{p, tf, mesh};
  
  dgSolver.dgTimeStep();
  
  return 0;
  
  // TODO: debugging
  p = 4;
  
  darray x3D, w3D;
  int n = gaussQuad3D(p, x3D, w3D);
  //std::cout << "3d points  = " << x3D << std::endl;
  //std::cout << "3d weights = " << w3D << std::endl;
  
  darray cheby3D = chebyshev3D(4);
  /*
  for (int j = 0; j < 5*5*5; ++j) {
    std::cout << "cheby[" << j << "] = {" 
	      << cheby3D(0,j) << ", " << cheby3D(1,j) << ", " << cheby3D(2,j) << "}" << std::endl;
  }
  */
  
  darray cheby = chebyshev(p);
  darray legend = legendre(p, cheby);
  darray dlegend = dlegendre(p, cheby);
  
  /*
  std::cout << std::endl << "legend = " << std::endl;
  for (int i = 0; i < legend.size(1); ++i) {
    for (int j = 0; j < legend.size(0); ++j) {
      std::cout << legend(i,j) << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl << "dlegend = " << std::endl;
  for (int i = 0; i < dlegend.size(1); ++i) {
    for (int j = 0; j < dlegend.size(0); ++j) {
      std::cout << dlegend(i,j) << ", ";
    }
    std::cout << std::endl;
  }
  */
  
  return 0;
  
  /** Tests Lapack(e) to ensure it can solve example */
  p = 4;
  
  // Initialize V to be specific vandermonde matrix
  darray V{p+1,p+1}; // also allocates memory
  for (int i = 0; i < V.size(0); ++i) {
    V(i, 0) = 1.0;
  }
  for (int j = 0; j < V.size(1); ++j) {
    V(p, j) = 1.0;
  }
  V(0,1) = -1.0; V(1,1) = -sqrt(2.0)/2.0; V(2,1) = 0; V(3,1) = sqrt(2.0)/2.0;
  V(0,2) = 1.0; V(1,2) = .25; V(2,2) = -0.5; V(3,2) = .25;
  V(0,3) = -1.0; V(1,3) = sqrt(2.0)/8.0; V(2,3) = 0; V(3,3) = -sqrt(2.0)/8.0;
  V(0,4) = 1.0; V(1,4) = -.40625; V(2,4) = .375; V(3,4) = -.40625;
  std::cout << V << std::endl;
  
  // Initialize RHSs to be identity matrix
  darray RHSs{p+1,p+1};
  for (int i = 0; i < p+1; ++i) {
    RHSs(i,i) = 1.0;
  }
  std::cout << "I = " << std::endl << RHSs << std::endl;
  
  // Solve system of equations
  MKL_INT ipiv[p+1];
  int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, p+1, p+1, V.data(), p+1, ipiv, RHSs.data(), p+1);
  std::cout << "output = " << std::endl << RHSs << std::endl;
  for (int i = 0; i < p+1; ++i) {
    std::cout << "ipiv[" << i << "] = " << ipiv[i] << std::endl;
  }
  
  return 0;
}
