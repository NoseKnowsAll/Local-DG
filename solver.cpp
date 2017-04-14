#include "solver.h"

/** Default constructor */
Solver::Solver() : Solver{2, 1.0, Mesh{}} { }

/** Main constructor */
Solver::Solver(int _p, double _tf, const Mesh& _mesh) :
  mesh{_mesh},
  tf{_tf},
  dt{},
  timesteps{},
  order{_p},
  dofs{},
  refNodes{},
  Mel{},
  Sels{},
  Kels{},
  u{}
{
  
  // Initialize time stepping information
  // TODO: choose dt based on CFL condition - DEPENDS ON PDE
  double vel = 1.0;
  dt = 0.5*std::min(std::min(mesh.minDX, mesh.minDY), mesh.minDZ)/vel;
  // Ensure we will exactly end at tf
  timesteps = std::ceil(tf/dt);
  dt = tf/timesteps;
  
  // Initialize nodes within elements
  refNodes = chebyshev3D(order);
  dofs = refNodes.size(1)*refNodes.size(2)*refNodes.size(3);
  mesh.setupNodes(refNodes, order);
  
  // Compute local matrices
  precomputeLocalMatrices();
  
  // TODO: does initial condition go here?
  
}

/** Precomputes all the local matrices used by Local DG method */
void Solver::precomputeLocalMatrices() {
  
  // Create nodal representation of the reference bases
  darray l3D = legendre3D(order, refNodes);
  
  darray coeffsPhi{order+1,order+1,order+1,order+1,order+1,order+1};
  for (int ipz = 0; ipz <= order; ++ipz) {
    for (int ipy = 0; ipy <= order; ++ipy) {
      for (int ipx = 0; ipx <= order; ++ipx) {
	coeffsPhi(ipx,ipy,ipz,ipx,ipy,ipz) = 1.0;
      }
    }
  }
  MKL_INT ipiv[dofs];
  int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, dofs, dofs, 
			   l3D.data(), dofs, ipiv, coeffsPhi.data(), dofs);
  
  // Compute reference bases on the quadrature points
  darray xQ, wQ;
  int sizeQ = gaussQuad3D(2*order, xQ, wQ);
  darray polyQuad = legendre3D(order, xQ);
  darray dPolyQuad = dlegendre3D(order, xQ);
  
  darray phiQ{sizeQ,order+1,order+1,order+1};
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
	      sizeQ, dofs, dofs, 1.0, polyQuad.data(), sizeQ, 
	      coeffsPhi.data(), dofs, 0.0, phiQ.data(), sizeQ);
  darray dPhiQ{sizeQ,order+1,order+1,order+1,Mesh::DIM};
  for (int l = 0; l < Mesh::DIM; ++l) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
		sizeQ, dofs, dofs, 1.0, &dPolyQuad(0,0,0,0,l), sizeQ, 
		coeffsPhi.data(), dofs, 0.0, &dPhiQ(0,0,0,0,l), sizeQ);
  }
  
  // Store weights*phiQ in polyQuad to avoid recomputing 4 times
  polyQuad = phiQ;
  for (int ipz = 0; ipz <= order; ++ipz) {
    for (int ipy = 0; ipy <= order; ++ipy) {
      for (int ipx = 0; ipx <= order; ++ipx) {
	for (int iQ = 0; iQ < sizeQ; ++iQ) {
	  polyQuad(iQ, ipx,ipy,ipz) *= wQ(iQ);
	}
      }
    }
  }
  
  // Initialize mass matrix = integrate(phi_i*phi_j)
  Mel.realloc(dofs,dofs);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
	      dofs, dofs, sizeQ, 1.0, phiQ.data(), sizeQ, 
	      polyQuad.data(), sizeQ, 0.0, Mel.data(), dofs);
  
  // Initialize stiffness matrices = integrate(dx_l(phi_i)*phi_j)
  Sels.realloc(dofs,dofs,Mesh::DIM);
  for (int l = 0; l < Mesh::DIM; ++l) {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
		dofs, dofs, sizeQ, 1.0, &dPhiQ(0,0,0,0,l), sizeQ, 
		polyQuad.data(), sizeQ, 0.0, &Sels(0,0,l), dofs);
  }
  
  // Initialize K matrices = dx_l(phi_i)*weights
  Kels.realloc(dofs,dofs,Mesh::DIM);
  for (int l = 0; l < Mesh::DIM; ++l) {
    for (int ipz = 0; ipz <= order; ++ipz) {
      for (int ipy = 0; ipy <= order; ++ipy) {
	for (int ipx = 0; ipx <= order; ++ipx) {
	  for (int iQ = 0; iQ < sizeQ; ++iQ) {
	    Kels(iQ, ipx,ipy,ipz, l) = dPhiQ(iQ, ipx,ipy,ipz, l)*wQ(iQ);
	  }
	}
      }
    }
  }
  
}

// TODO: set u based off of some function?
void Solver::initialCondition() {
  
}

/**
   Actually time step Local DG method according to RK4, 
   updating the solution u every time step
*/
void Solver::dgTimeStep() {
  
}
