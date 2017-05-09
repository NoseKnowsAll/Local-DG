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
  
  /* For convection-diffusion 
  typedef struct {
    const double a[Mesh::DIM] = {1,2,3};
    const double eps = 1e-2;
  } physics;
  */
  
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
  } physics;
  
  physics p;
  
  void precomputeLocalMatrices();
  void precomputeInterpMatrices();
  void initialCondition();
  void trueSolution(darray& uTrue, double t) const;
  
  void rk4UpdateCurr(darray& uCurr, const darray& diagA, const darray& ks, int istage) const;
  void rk4Rhs(const darray& uCurr, darray& uInterp2D, darray& uInterp3D, 
	      darray& Dus, darray& DuInterp2D, darray& DuInterp3D, 
	      darray& toSend, darray& toRecv, MPI_Request * rk4Reqs, darray& ks, int istage) const;
  
  void interpolate(const darray& curr, darray& toInterp2D, darray& toInterp3D,
		   darray& toSend, darray& toRecv, MPI_Request * rk4Reqs, int dim) const;
  
  void localDGFlux(const darray& uInterp2D, darray& residuals) const;
  
  void convectDGFlux(const darray& uInterp2D, darray& residual) const;
  void convectDGVolume(const darray& uInterp3D, darray& residual) const;
  
  void viscousDGFlux(const darray& uInterp2D, const darray& DuInterp2D, darray& residual) const;
  void viscousDGVolume(const darray& uInterp3D, const darray& DuInterp3D, darray& residual) const;
  
  
  inline double numericalFluxL(double uK, double uN, double normalK) const;
  void numericalFluxC(const darray& uN, const darray& uK, 
		      const darray& normalK, darray& fluxes) const;
  void numericalFluxV(const darray& uN, const darray& uK, 
		      const darray& DuN, const darray& DuK, 
		      const darray& normalK, darray& fluxes) const;
  
  inline void fluxC(const darray& uK, darray& fluxes) const;
  inline void fluxV(const darray& uK, const darray& DuK, darray& fluxes) const;
  
  
  void mpiStartComm(const darray& interpolated, int dim, darray& toSend, darray& toRecv, MPI_Request * rk4Reqs) const;
  void mpiEndComm(darray& interpolated, int dim, const darray& toRecv, MPI_Request * rk4Reqs) const;
  
  double computeKE(const darray& uInterp3D) const;
  
public:
  Solver();
  Solver(int _p, int _dtSnaps, double _tf, double _L, const Mesh& _mesh);
  
  void dgTimeStep();
  
};

#endif
