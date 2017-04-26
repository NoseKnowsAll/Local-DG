#include <iostream>
#include <mkl_lapacke.h>
#include <cmath>

#include "dgMath.h"
#include "mesh.h"
#include "solver.h"
#include "io.h"

/** Main driver function */
int main(int argc, char *argv[]) {
  
  //Point botLeft{-1,-1,-1};
  //Point topRight{1,1,1};
  //Mesh mesh{1,1,1, botLeft, topRight};
  
  Mesh mesh{};
  
  int p = 2;
  double tf = 1.0;
  Solver dgSolver{p, tf, mesh};
  
  dgSolver.dgTimeStep();
  
  return 0;
}
