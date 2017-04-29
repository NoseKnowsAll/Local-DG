#include <iostream>
#include <mkl_lapacke.h>
#include <cmath>
#include <mpi.h>

#include "dgMath.h"
#include "mesh.h"
#include "solver.h"
#include "io.h"

/** Main driver function */
int main(int argc, char *argv[]) {
  
  MPI_Init(&argc, &argv);
  MPIUtil mpi{};
  
  std::cout << "initialized MPIUtil" << std::endl;
  
  //Point botLeft{-1,-1,-1};
  //Point topRight{1,1,1};
  //Mesh mesh{1,1,1, botLeft, topRight};
  
  Mesh mesh{mpi};
  
  std::cout << "initialized mesh" << std::endl;
  
  int p = 2;
  double tf = 1.0;
  int dtSnaps = 30;
  Solver dgSolver{p, dtSnaps, tf, mesh};
  
  std::cout << "initialized solver" << std::endl;
  
  dgSolver.dgTimeStep();
  
  MPI_Finalize();
  return 0;
}
