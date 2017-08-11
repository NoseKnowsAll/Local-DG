#ifndef SOLVER_H__
#define SOLVER_H__

#include <iostream>
#ifdef __INTEL_COMPILER
#include <mkl_lapacke.h>
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
#include <cmath>
#include <chrono>

#include "source.h"
#include "mesh.h"
#include "io.h"
#include "MPIUtil.h"
#include "dgMath.h"
#include "rk4.h"
#include "array.h"

class Solver {
private:
  
  Mesh mesh;
  MPIUtil mpi;
  
  double tf;
  double dt;
  dgSize timesteps;
  dgSize stepsPerSnap;
  
  int order;
  int dofs;
  darray refNodes;
  int dofsF;
  darray refNodesF;
  
  int nQV;
  darray xQV;
  darray wQV;
  int nQF;
  darray xQF;
  darray wQF;
  
  int nStates;
  darray Mel;
  iarray Mipiv;
  iarray sToS;
  darray Kels;
  darray KelsF;
  
  darray InterpF;   // Interpolates faces nodes to face Gaussian pts
  darray InterpV;   // Interpolates volume nodes to volume Gaussian pts
  darray InterpW;   // InterpV*Weights
  darray InterpTk;  // Interpolates bilinear map Tk to volume nodes
  darray InterpTkQ; // Interpolates bilinear map Tk to volume quadrature pts
  
  darray Jk;  // |det(Jacobian(Tk))| at each volume quadrature point
  darray JkF; // |det(Jacobian(Tk))| at each face quadrature point
  
  darray u;
  
  // For Elastic
  typedef struct {
    // Use reasonable constants when no input file
    const double vpConst = std::sqrt((2.2+2*1.3)/1.2);//200.0;
    const double vsConst = std::sqrt(1.3/1.2);//80.0;
    const double rhoConst = 1.2;
    double C;      // max velocity throughout domain
    darray lambda; // Lame's first parameter
    darray mu;     // Lame's second parameter
    darray rho;    // density
    Source src;    // forcing term
  } physics;
  
  /* // For convection-diffusion 
  typedef struct {
    const double a[Mesh::DIM] = {1,2};
    const double eps = 1e-2;
  } physics; */
  
  physics p;
  
  void precomputeLocalMatrices();
  void precomputeInterpMatrices();
  
  void initMaterialProps();
  void initialCondition();
  void initTimeStepping(double dtSnap);
  void initSource(Source::Params& srcParams);
  
  void trueSolution(darray& uTrue, double t) const;
  void computePressure(const darray& uInterpV, darray& pressure) const;
  
  void rk4UpdateCurr(darray& uCurr, const darray& diagA, const darray& ks, int istage) const;
  void rk4Rhs(const darray& uCurr, darray& uInterpF, darray& uInterpV, 
	      darray& toSend, darray& toRecv, MPI_Request * rk4Reqs, 
	      darray& ks, int istage, int iTime) const;
  
  void interpolate(const darray& curr, darray& toInterpF, darray& toInterpV,
		   darray& toSend, darray& toRecv, MPI_Request * rk4Reqs, int dim) const;
  
  void sourceVolume(darray& residual, int istage, int iTime) const;
  
  void convectDGFlux(const darray& uInterpF, darray& residual) const;
  void convectDGVolume(const darray& uInterpV, darray& residual) const;
  
  void boundaryFluxC(const darray& uK, const darray& normalK, darray& fluxes, 
		     Mesh::Boundary bc, double lambdaK, double muK, double rhoK) const;
  void numericalFluxC(const darray& uN, const darray& uK, 
		      const darray& normalK, darray& fluxes,
		      double lambdaN, double muN, double rhoN, 
		      double lambdaK, double muK, double rhoK) const;
  
  inline void fluxC(const darray& uK, darray& fluxes, double lambda, double mu) const;
  
  double computeL2Error(const darray& uCurr, darray& uTrue) const;
  double computeL2Norm(const darray& uCurr) const;
  
  double computeInfError(const darray& uCurr, darray& uTrue) const;
  double computeInfNorm(const darray& uCurr) const;
  
  void mpiStartComm(const darray& interpolated, int dim, darray& toSend, darray& toRecv, MPI_Request * rk4Reqs) const;
  void mpiEndComm(darray& interpolated, int dim, const darray& toRecv, MPI_Request * rk4Reqs) const;
  void mpiSendMaterials();
  
public:
  Solver();
  Solver(int _order, Source::Params srcParams, double dtSnap, double _tf, const Mesh& _mesh);
  
  void dgTimeStep();
  
};

#endif
