#ifndef SOLVER_H__
#define SOLVER_H__

#include <iostream>
#include <mkl_lapacke.h>
#include <mkl.h>
#include <cmath>
#include <chrono>

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
  int nStates;
  darray Mel;
  darray Sels;
  darray Kels;
  
  darray wQ2D;
  darray Interp2D;
  darray wQ3D;
  darray Interp3D;
  
  darray u;
  
  void precomputeLocalMatrices();
  void precomputeInterpMatrices();
  
  void rhs(const darray& ucurr, darray& ks, int istage) const;
  
  darray localDGFlux(const darray& u) const;
  inline double fluxL(double u) const;
  
public:
  Solver();
  Solver(int _p, double _tf, const Mesh& _mesh);
  
  void initialCondition();
  void dgTimeStep();

};

#endif
