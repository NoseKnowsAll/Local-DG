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
  
  int p = 2;
  double tf = 2.0;
  Solver dgSolver{p, tf, mesh};
  
  dgSolver.dgTimeStep();
  
  return 0;
}
