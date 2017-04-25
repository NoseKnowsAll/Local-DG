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
  darray Kels2D;
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
  void rk4Rhs(const darray& uCurr, darray& uInterp2D, darray& uInterp3D, 
	      darray& Dus, darray& DuInterp2D, darray& DuInterp3D, 
	      darray& ks, int istage) const;
  
  void interpolateU(const darray& uCurr, darray& uInterp2D, darray& uInterp3D) const;
  void interpolateDu(const darray& Dus, darray& DuInterp2D, darray& DuInterp3D) const;
  
  void localDGFlux(const darray& uInterp2D, darray& residuals) const;
  inline double numericalFluxL(double uK, double uN, 
			       double normalK, double normalN) const;
  
  void convectDGFlux(const darray& uInterp2D, darray& residual) const;
  void convectDGVolume(const darray& uInterp3D, darray& residual) const;
  inline double numericalFluxC(double uK, double uN, 
			       const darray& normalK, const darray& normalN) const;
  inline double fluxC(double uK, int l) const;
  
  void viscousDGFlux(const darray& uInterp2D, const darray& DuInterp2D, darray& residual) const;
  void viscousDGVolume(const darray& uInterp3D, const darray& DuInterp3D, darray& residual) const;
  inline double numericalFluxV(double uK, double uN, const darray& DuK, const darray& DuN, 
			       const darray& normalK, const darray& normalN) const;
  inline double fluxV(double uK, const darray& DuK, int l) const;
  
  
public:
  Solver();
  Solver(int _p, double _tf, const Mesh& _mesh);
  
  void dgTimeStep();
  
};

#endif
