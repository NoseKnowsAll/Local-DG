#ifndef SOLVER_H__
#define SOLVER_H__

#include <iostream>
#include <mkl_lapacke.h>
#include <mkl.h>
#include <cmath>

#include "mesh.h"
#include "dgMath.h"

class Solver {
private:
  
  Mesh mesh;
  
  double tf;
  double dt;
  long timesteps;
  
  int order;
  int dofs;
  darray refNodes;
  darray Mel;
  darray Sels;
  darray Kels;
  
  darray u;
  
  void precomputeLocalMatrices();
  
public:
  Solver();
  Solver(int _p, double _tf, const Mesh& _mesh);
  
  void initialCondition();
  void dgTimeStep();

};

#endif
