#include <iostream>
#include <mkl_lapacke.h>
#include <cmath>

#include "mesh.h"
#include "solver.h"
#include "io.h"

/** Main driver function */
int main(int argc, char *argv[]) {
  
  /* Clear output debug fileds
  clearSSVFile("output/x.txt");
  clearSSVFile("output/y.txt");
  clearSSVFile("output/z.txt");
  clearSSVFile("output/u.txt");
  */
  initXYZVFile("output/xyzu.txt", "u");
  
  Mesh mesh{};
  //std::cout << mesh << std::endl;
  
  int p = 2;
  double tf = 2.0;
  Solver dgSolver{p, tf, mesh};
  
  dgSolver.dgTimeStep();
  
  return 0;
}
