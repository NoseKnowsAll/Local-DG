#ifndef SOLVER_H__
#define SOLVER_H__

#include <iostream>
#include <mkl_lapacke.h>
#include <mkl.h>
#include <cmath>
#include <chrono>

#include "mesh.h"
#include "MPIUtil.h"
#include "dgMath.h"
#include "io.h"

class Solver {
private:
  
  Mesh mesh;
  MPIUtil mpi;
  
  double tf;
  double dt;
  long timesteps;
  int dtSnaps;
  
  int order;
  int dofs;
  darray refNodes;
  int nStates;
  darray Mel;
  iarray Mipiv;
  darray Sels;
  darray Kels;
  darray Kels2D;
  
  darray Interp2D;
  darray Interp3D;
  
  darray u;
  darray u0;
  
  const double a[Mesh::DIM];
  
  void precomputeLocalMatrices();
  void precomputeInterpMatrices();
  void initialCondition();
  void trueSolution(double t);
  
  void rk4UpdateCurr(darray& uCurr, const darray& diagA, const darray& ks, int istage) const;
  void rk4Rhs(const darray& uCurr, darray& uInterp2D, darray& uInterp3D, 
	      darray& Dus, darray& DuInterp2D, darray& DuInterp3D, 
	      darray& uToSend, darray& uToRecv, darray& DuToSend, darray& DuToRecv,
	      darray& ks, int istage) const;
  
  void interpolateU(const darray& uCurr, darray& uInterp2D, darray& uInterp3D,
		    darray& uToSend, darray& uToRecv) const;
  void interpolateDus(const darray& Dus, darray& DuInterp2D, darray& DuInterp3D) const;
  
  void localDGFlux(const darray& uInterp2D, darray& residuals) const;
  inline double numericalFluxL(double uK, double uN, double normalK) const;
  
  void convectDGFlux(const darray& uInterp2D, darray& residual) const;
  void convectDGVolume(const darray& uInterp3D, darray& residual) const;
  inline double numericalFluxC(double uK, double uN, const darray& normalK) const;
  inline double fluxC(double uK, int l) const;
  
  void viscousDGFlux(const darray& uInterp2D, const darray& DuInterp2D, darray& residual) const;
  void viscousDGVolume(const darray& uInterp3D, const darray& DuInterp3D, darray& residual) const;
  inline double numericalFluxV(double uK, double uN, const darray& DuK, const darray& DuN, 
			       const darray& normalK) const;
  inline double fluxV(double uK, const darray& DuK, int l) const;
  
  void mpiStartComm(const darray& interpolated, int dim, darray& toSend, darray& toRecv, MPI_Request * rk4Reqs) const;
  void mpiEndComm(darray& interpolated, int dim, const darray& toRecv, MPI_Request * rk4Reqs) const;
  
public:
  Solver();
  Solver(int _p, int _dtSnaps, double _tf, const Mesh& _mesh);
  
  void dgTimeStep();
  
};

#endif
