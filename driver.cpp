#include <iostream>
#include <cmath>
#include <mpi.h>

#include "solver.h"
#include "source.h"
#include "mesh.h"

/** Main driver function */
int main(int argc, char *argv[]) {
  
  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPIUtil mpi{};
  if (mpi.rank == mpi.ROOT) {
    std::cout << "MPI tasks = " << mpi.np << std::endl;
  }
  
  // Initialize mesh
  if (mpi.rank == mpi.ROOT) {
    std::cout << "Initializing mesh..." << std::endl;
  }
#define USING_MESH 1
#ifdef USING_MESH
  std::string mshFile{"/global/homes/f/frms4q/repos/Summer2017/Local-DG/input/unitSquare.msh"};
  //std::string mshFile{"/global/homes/f/frms4q/repos/Summer2017/Local-DG/input/mySquareX.msh"};
  Mesh mesh{mshFile, mpi};
#else
  /*
  Point botLeft{50.0, 25.0};
  Point topRight{200.0, 100.0};
  int nx = 15;
  Mesh mesh{2*nx, nx, botLeft, topRight, mpi};
  */
  Point botLeft{0.0, 0.0};
  Point topRight{1.0, 1.0};
  int nx = 10;
  Mesh mesh{nx, nx, botLeft, topRight, mpi};
  
  std::string outputFile{"/global/homes/f/frms4q/repos/Summer2017/Local-DG/output/mySquare.msh"};
  mesh.outputMesh(outputFile, nx, nx);
#endif
  
  if (mpi.rank == mpi.ROOT) {
    std::cout << "dx: min = " << mesh.dxMin << ", max = " << mesh.dxMax << std::endl;
  }
  
  // Initialize sources
  int nsrcs = 1;
  Source::Params srcParams;
  srcParams.srcPos.realloc(Mesh::DIM, nsrcs);
  srcParams.srcPos(0,0) = 150.0;
  srcParams.srcPos(1,0) = 50.0;
  srcParams.srcAmps.realloc(nsrcs);
  srcParams.srcAmps(0) = 0.0;
  srcParams.type = Source::Wavelet::null;
  //srcParams.maxF = 10.0;
  
  // Initialize solver
  if (mpi.rank == mpi.ROOT) {
    std::cout << "Initializing solver..." << std::endl;
  }
  int order = 4;
  double tf = 0.1;
  double dtSnap = 0.005;
  
  if (mpi.rank == mpi.ROOT) {
    std::cout << "order = " << order << std::endl;
    std::cout << "tf = " << tf << std::endl;
#ifdef USING_MESH
    std::cout << "Loaded mesh = " << mshFile << std::endl;
#else
    std::cout << "square domain: " << botLeft << ", " << topRight << std::endl;
    std::cout << "nx = " << nx << std::endl;
#endif
  }
  Solver dgSolver{order, srcParams, dtSnap, tf, mesh};
  
  // Run time steps
  dgSolver.dgTimeStep();
  
  MPI_Finalize();
  return 0;
}
