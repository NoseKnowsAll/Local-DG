#include <iostream>
#include <mkl_lapacke.h>
#include <cmath>

#include "mesh.h"
#include "solver.h"
#include "io.h"

/** Main driver function */
int main(int argc, char *argv[]) {
  
  Mesh mesh{};
  //std::cout << mesh << std::endl;
  
  int p = 2;
  double tf = 1.0;
  Solver dgSolver{p, tf, mesh};
  
  dgSolver.dgTimeStep();
  
  return 0;
}
