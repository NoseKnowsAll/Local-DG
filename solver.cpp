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
  nStates{},
  Mel{},
  Sels{},
  Kels{},
  wQ2D{},
  Interp2D{},
  wQ3D{},
  Interp3D{},
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
  
  // TODO: DEPENDS ON PDE
  nStates = 1;
  u.realloc(dofs, nStates, mesh.nElements);
  initialCondition();
  
  // Compute local matrices
  precomputeLocalMatrices();
  
  // Compute interpolation matrices
  precomputeInterpMatrices();
  
    
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
  
  // TODO: Should we also precompute inv(Mel) right here? 
  
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

/** Precomputes all the interpolation matrices used by Local DG method */
void Solver::precomputeInterpMatrices() {
  
  // 2D interpolation matrix for use on faces
  darray cheby2D = chebyshev2D(order);
  darray xQ2D;
  int nquads2D   = gaussQuad2D(order, xQ2D, wQ2D);
  
  Interp2D = interpolationMatrix2D(cheby2D, xQ2D);
  
  // 3D interpolation matrix for use on elements
  darray cheby3D = chebyshev3D(order);
  darray xQ3D;
  int nquads3D   = gaussQuad3D(order, xQ3D, wQ3D);
  
  Interp3D = interpolationMatrix3D(cheby3D, xQ3D);
  
}

// TODO: set u based off of some function?
void Solver::initialCondition() {
  
}

/**
   Actually time step Local DG method according to RK4, 
   updating the solution u every time step
*/
void Solver::dgTimeStep() {
  
  // Define RK4
  int nStages = 4;
  darray c{nStages};
  c(0) = 0.0;
  c(1) = 0.5;
  c(2) = 0.5;
  c(3) = 1.0;
  
  darray b{nStages};
  b(0) = 1/6.0;
  b(1) = 1/3.0;
  b(2) = 1/3.0;
  b(3) = 1/6.0;
  
  // explicit & diagonal => stored as off diagonal vector
  darray diagA{nStages-1};
  diagA(0) = 0.5;
  diagA(1) = 0.5;
  diagA(2) = 1.0;
  
  darray uTemp{dofs, nStates, mesh.nElements};
  darray ks{dofs, nStates, mesh.nElements, nStages};
  
  auto startTime = std::chrono::high_resolution_clock::now();
  
  // Loop over time steps
  for (int iStep = 0; iStep < timesteps; ++iStep) {
    printf("time = %f\n", iStep*dt);
    
    // Use RK4 to update k values
    for (int istage = 0; istage < nStages; ++istage) {
      
      if (istage == 0) {
	for (int iK = 0; iK < mesh.nElements; ++iK) {
	  for (int iS = 0; iS < nStates; ++iS) {
	    for (int iN = 0; iN < dofs; ++iN) {
	      uTemp(iN,iS,iK) = u(iN,iS,iK);
	    }
	  }
	}
      }
      else {
	for (int iK = 0; iK < mesh.nElements; ++iK) {
	  for (int iS = 0; iS < nStates; ++iS) {
	    for (int iN = 0; iN < dofs; ++iN) {
	      uTemp(iN,iS,iK) = u(iN,iS,iK) + dt*diagA(istage-1)*ks(iN,iS,iK,istage-1);
	    }
	  }
	}
      }
      
      // Updates ks(:,istage) based on DG method
      rhs(uTemp, ks, istage);
      
    }
    
    // Use RK4 to move to next time step
    for (int istage = 0; istage < nStages; ++istage) {
      for (int iK = 0; iK < mesh.nElements; ++iK) {
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iN = 0; iN < dofs; ++iN) {
	    u(iN,iS,iK) += dt*b(istage)*ks(iN,iS,iK,istage);
	  }
	}
      }
    }
    
  }
  
  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = endTime-startTime;
  std::cout << "Finished time stepping. Time elapsed = " << elapsed.count() << std::endl;
  
}

/**
   Computes the RHS of Runge Kutta method for use with Local DG method.
   Updates ks(:,istage) with rhs evaluated at ucurr
*/
void Solver::rhs(const darray& ucurr, darray& ks, int istage) const {
  
  
  
}


/**
   Local DG Flux: Computes Fl(u) for use in the local DG formulation of second terms.
   Uses a Lax-Friedrichs formulation for the flux term. TODO: Shouldn't we be using downwind?
*/
darray Solver::localDGFlux(const darray& u) const {
  
  int nQ2D = wQ2D.size(0);
  darray globalFlux{dofs, nStates, mesh.nElements, Mesh::DIM};
  
  // Loop over all elements
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    
    // Initialize flux along faces
    darray fStar{mesh.nFNodes, Mesh::N_FACES, nStates, Mesh::DIM}; // TODO: should we move this out of for loop
    darray faceContributions{mesh.nFNodes, Mesh::N_FACES, nStates, Mesh::DIM};
    
    for (int l = 0; l < Mesh::DIM; ++l) {
      for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
	
	auto nF = mesh.eToF(iF, iK);
	auto nK = mesh.eToE(iF, iK);
	
	auto normalK = mesh.normals(l, iF, iK);
	auto normalN = mesh.normals(l, nF, nK);
	
	for (int iN = 0; iN < nStates; ++iN) {
	  
	  // For every face, compute flux = integrate over face (fstar*phi_i)
	  // Must compute nStates of these flux integrals per face
	  darray integrand{Interp2D};
	  
	  for (int iFN = 0; iFN < mesh.nFNodes; ++iFN) {
	    auto uK = u(mesh.efToN(iFN, iF), iN, iK);
	    auto uN = u(mesh.efToN(iFN, nF), iN, nK);
	    auto fK = fluxL(uK);
	    auto fN = fluxL(uN);
	    // TODO: This appears to also be the Roe A value?
	    double C = std::abs((fN-fK)/(uN-uK));
	    
	    fStar(iFN, iF, iN, l) = (fK+fN)/2.0 + 
	      (C/2.0)*(uN*normalN + uK*normalK);
	  }
	  
	  // Flux integrand = Interp2D*diag(fstar)
	  for (int iFN = 0; iFN < mesh.nFNodes; ++iFN) {
	    for (int iQ = 0; iQ < nQ2D; ++iQ) {
	      integrand(iQ, iFN) *= fStar(iFN, iF, iN, l);
	    }
	  }
	  
	  // contribution = integrand'*wQ2D
	  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
		      mesh.nFNodes, 1, nQ2D, 1.0, integrand.data(), nQ2D, 
		      wQ2D.data(), nQ2D, 
		      0.0, &faceContributions(0,iF,iN,l), mesh.nFNodes);
	  
	}
	
      }
    }
    
    // Add up face contributions into global flux array
    // Performed outside the above loops, thinking ahead to OpenMP...
    // TODO: globalFlux should be sparse, should we instead add directly into result?
    for (int l = 0; l < Mesh::DIM; ++l) {
      for (int iN = 0; iN < nStates; ++iN) {
	for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
	  for (int iFN = 0; iFN < mesh.nFNodes; ++iFN) {
	    globalFlux(mesh.efToN(iFN,iF), iN, iK, l) += faceContributions(iFN, iF, iN, l);
	  }
	}
      }
    }
    
    
  }
  
  return globalFlux;
}

/** Evaluates the local DG flux function for this PDE at a given value u */
inline double Solver::fluxL(double u) const {
  return u;
}
