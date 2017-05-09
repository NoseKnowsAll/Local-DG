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
  
  if (mpi.rank == mpi.ROOT) {
    std::cout << "MPI tasks = " << mpi.np << std::endl;
  }
  /*
  std::ostringstream oss;
  oss << "coords = " << mpi.coords;
  mpi.printString(oss.str());
  */
  
  if (mpi.rank == mpi.ROOT) {
    std::cout << "Initializing mesh..." << std::endl;
  }
  
  //Mesh mesh{mpi};
  double L = 1;
  double size = M_PI*L;
  Point botLeft{-size, -size, -size};
  Point topRight{size, size, size};
  int nx = 64;
  Mesh mesh{nx, nx, nx, botLeft, topRight, mpi};
  
  if (mpi.rank == mpi.ROOT) {
    std::cout << "Initializing solver..." << std::endl;
  }
  
  int p = 2;
  double tf = 20.0;
  int dtSnaps = 2000000; // TODO
  if (mpi.rank == mpi.ROOT) {
    std::cout << "p = " << p << std::endl;
    std::cout << "tf = " << tf << std::endl;
    std::cout << "nx = " << nx << std::endl;
  }
  Solver dgSolver{p, dtSnaps, tf, L, mesh};
  
  dgSolver.dgTimeStep();
  
  MPI_Finalize();
  return 0;
}
