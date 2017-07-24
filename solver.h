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
#include "MPIUtil.h"
#include "dgMath.h"
#include "io.h"
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
  int nQV;
  darray refNodes;
  darray xQV;
  int nStates;
  darray Mel;
  iarray Mipiv;
  iarray sToS;
  darray Kels;
  darray KelsF;
  
  darray InterpF;
  darray InterpV;
  darray InterpW;
  
  darray u;
  
  // For Elastic
  typedef struct {
    // Use reasonable constants when no input file
    const double vpConst = 200.0;
    const double vsConst = 80.0;
    const double rhoConst = 1.0;
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

  /* // For Navier-Stokes
  typedef struct {
    double pars[2]; // Reynolds number, Prandtl number
    double gamma;   // adiabatic gas constant = ratio of specific heats = 1.4 in ideal gas
    double M0;      // background Mach number
    double rho0;    // background density
    double V0;      // background velocity
    double L;       // size of domain = pi*L
    //double mu;      // dynamic shear viscosity
    //double kappa;   // heat conductivity
    double c0;      // background speed of sound
    double p0;      // background pressure
    double T0;      // background temperature
    double tc;      // characteristic convective time
    double R;       // ideal gas constant
  } physics; */
  
  physics p;
  
  void precomputeLocalMatrices();
  void precomputeInterpMatrices();
  
  void initTimeStepping(double dtSnap);
  void initMaterialProps();
  void initialCondition();
  void trueSolution(darray& uTrue, double t) const;
  void computePressure(const darray& uInterpV, darray& pressure) const;
  
  void rk4UpdateCurr(darray& uCurr, const darray& diagA, const darray& ks, int istage) const;
  void rk4Rhs(const darray& uCurr, darray& uInterpF, darray& uInterpV, 
	      darray& toSend, darray& toRecv, MPI_Request * rk4Reqs, 
	      darray& ks, int istage, int iTime) const;
  
  void interpolate(const darray& curr, darray& toInterpF, darray& toInterpV,
		   darray& toSend, darray& toRecv, MPI_Request * rk4Reqs, int dim) const;
  
  void sourceVolume(darray& residual, int iTime) const;
  
  void convectDGFlux(const darray& uInterpF, darray& residual) const;
  void convectDGVolume(const darray& uInterpV, darray& residual) const;
  
  void numericalFluxC(const darray& uN, const darray& uK, 
		      const darray& normalK, darray& fluxes,
		      double lambdaN, double muN, double rhoN, 
		      double lambdaK, double muK, double rhoK) const;
  
  inline void fluxC(const darray& uK, darray& fluxes, double lambda, double mu) const;
  
  void mpiStartComm(const darray& interpolated, int dim, darray& toSend, darray& toRecv, MPI_Request * rk4Reqs) const;
  void mpiEndComm(darray& interpolated, int dim, const darray& toRecv, MPI_Request * rk4Reqs) const;
  void mpiSendMaterials();
  
public:
  Solver();
  Solver(int _order, Source::Params srcParams, double dtSnap, double _tf, const Mesh& _mesh);
  
  void dgTimeStep();
  
};

#endif
