#ifndef SOLVER_H__
#define SOLVER_H__

#include <iostream>
#include <mkl_lapacke.h>
#include <mkl.h>
#include <cmath>
#include <chrono>

#include "mesh.h"
#include "dgMath.h"
#include "io.h"

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
  darray M2D;
  
  darray Interp2D;
  darray Interp3D;
  
  darray u;
  darray u0;
  
  const double a[Mesh::DIM];
  
  void precomputeLocalMatrices();
  void precomputeInterpMatrices();
  void initialCondition();
  
  void rk4UpdateCurr(darray& uCurr, const darray& diagA, const darray& ks, int istage) const;
  void rk4Rhs(const darray& ucurr, darray& Dus, darray& ks, int istage) const;
  
  void localDGFlux(const darray& uCurr, darray& globalFlux) const;
  inline double numericalFluxL(double uK, double uN, double normalK, double normalN) const;
  
  void convectDGFlux(const darray& uCurr, darray& globalFlux) const;
  inline double numericalFluxC(double uK, double uN, const darray& normalK, const darray& normalN) const;
  inline double fluxC(double uK, int l) const;
  
  void convectDGVolume(const darray& uCurr, darray& residual) const;
  inline double fluxV(double uK, const darray& DuK, int l) const;
  void viscousDGVolume(const darray& uCurr, const darray& Dus, darray& residual) const;
  
public:
  Solver();
  Solver(int _p, double _tf, const Mesh& _mesh);
  
  void dgTimeStep();
  
};

#endif
