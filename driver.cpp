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
  
  std::ostringstream oss;
  oss << "coords = " << mpi.coords;
  mpi.printString(oss.str());
  
  if (mpi.rank == mpi.ROOT) {
    std::cout << "Initializing mesh..." << std::endl;
  }
  
  //Mesh mesh{mpi};
  Mesh mesh{10, 10, 10, Point{0.0,0.0,0.0}, Point{1.0,1.0,1.0}, mpi};
  
  if (mpi.rank == mpi.ROOT) {
    std::cout << "Initializing solver..." << std::endl;
  }
  
  int p = 2;
  double tf = 0.25;
  int dtSnaps = 30;
  Solver dgSolver{p, dtSnaps, tf, mesh};
  
  dgSolver.dgTimeStep();
  
  MPI_Finalize();
  return 0;
}
