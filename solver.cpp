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
  M2D{},
  Interp2D{},
  Interp3D{},
  u{},
  a{1,2,3}
{
  
  // Initialize time stepping information
  // TODO: choose dt based on CFL condition - DEPENDS ON PDE
  double maxVel = 3.0;
  dt = 0.5*std::min(std::min(mesh.minDX, mesh.minDY), mesh.minDZ)/maxVel;
  // Ensure we will exactly end at tf
  timesteps = std::ceil(tf/dt);
  dt = tf/timesteps;
  
  // Initialize nodes within elements
  chebyshev3D(order, refNodes);
  dofs = refNodes.size(1)*refNodes.size(2)*refNodes.size(3);
  mesh.setupNodes(refNodes, order);
  
  // TODO: DEPENDS ON PDE
  nStates = 1;
  u.realloc(dofs, nStates, mesh.nElements);
  initialCondition(); // sets u
  
  // Compute interpolation matrices
  precomputeInterpMatrices();
  
  // Compute local matrices
  precomputeLocalMatrices();
  
}

/** Precomputes all the interpolation matrices used by Local DG method */
void Solver::precomputeInterpMatrices() {
  
  // 2D interpolation matrix for use on faces
  darray cheby2D;
  chebyshev2D(order, cheby2D);
  darray xQ2D, wQ2D;
  int nquads2D   = gaussQuad2D(order, xQ2D, wQ2D);
  
  interpolationMatrix2D(cheby2D, xQ2D, Interp2D);
  
  // 3D interpolation matrix for use on elements
  darray cheby3D;
  chebyshev3D(order, cheby3D);
  darray xQ3D, wQ3D;
  int nquads3D   = gaussQuad3D(order, xQ3D, wQ3D);
  
  interpolationMatrix3D(cheby3D, xQ3D, Interp3D);
  
}

/** Precomputes all the local matrices used by Local DG method */
void Solver::precomputeLocalMatrices() {
  
  // Create nodal representation of the reference bases
  darray l3D;
  legendre3D(order, refNodes, l3D);
  
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
  darray polyQuad, dPolyQuad;
  legendre3D(order, xQ, polyQuad);
  dlegendre3D(order, xQ, dPolyQuad);
  
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
  // assumes all elements are the same size...
  darray alpha{Mesh::DIM};
  alpha(0) = mesh.minDX/2.0;
  alpha(1) = mesh.minDY/2.0;
  alpha(2) = mesh.minDZ/2.0;
  double Jacobian = alpha(0)*alpha(1)*alpha(2);
  Mel.realloc(dofs,dofs);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
	      dofs, dofs, sizeQ, Jacobian, phiQ.data(), sizeQ, 
	      polyQuad.data(), sizeQ, 0.0, Mel.data(), dofs);
  
  // TODO: We should also precompute inv(Mel) right here? 
  
  // Initialize stiffness matrices = integrate(dx_l(phi_i)*phi_j)
  Sels.realloc(dofs,dofs,Mesh::DIM);
  for (int l = 0; l < Mesh::DIM; ++l) {
    double scaleL = Jacobian/alpha(l);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
		dofs, dofs, sizeQ, scaleL, &dPhiQ(0,0,0,0,l), sizeQ, 
		polyQuad.data(), sizeQ, 0.0, &Sels(0,0,l), dofs);
  }
  
  // Initialize K matrices = dx_l(phi_i)*weights
  Kels.realloc(sizeQ,dofs,Mesh::DIM);
  for (int l = 0; l < Mesh::DIM; ++l) {
    double scaleL = Jacobian/alpha(l);
    for (int iDofs = 0; iDofs < dofs; ++iDofs) {
      for (int iQ = 0; iQ < sizeQ; ++iQ) {
	Kels(iQ, iDofs, l) = dPhiQ(iQ, iDofs,0,0, l)*wQ(iQ)*scaleL;
      }
    }
  }
  
  // Initialize 2D mass matrix for use along faces
  int fDofs = (order+1)*(order+1);
  darray weightedInterp{sizeQ,fDofs};
  for (int iDofs = 0; iDofs < fDofs; ++iDofs) {
    for (int iQ = 0; iQ < sizeQ; ++iQ) {
      weightedInterp(iQ,iDofs) = Interp2D(iQ,iDofs)*wQ(iQ);
    }
  }
  
  M2D.realloc(fDofs,fDofs,Mesh::DIM);
  for (int l = 0; l < Mesh::DIM; ++l) {
    double scaleL = Jacobian/alpha(l);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		fDofs,fDofs,sizeQ, scaleL, weightedInterp.data(), sizeQ,
		Interp2D.data(), sizeQ, 0.0, &M2D(0,0,l), fDofs);
  }
  
}

// TODO: Initialize u0 based off of a reasonable function
void Solver::initialCondition() {
  
  // Gaussian distribution with variance 0.2, centered around 0
  double sigmaSq = 0.2;
  double a = 1.0/(std::sqrt(sigmaSq*2*M_PI));
  
  for (int k = 0; k < mesh.nElements; ++k) {
    for (int iS = 0; iS < nStates; ++iS) {
      for (int iz = 0; iz < order+1; ++iz) {
	for (int iy = 0; iy < order+1; ++iy) {
	  for (int ix = 0; ix < order+1; ++ix) {
	    int vID = ix + iy*(order+1) + iz*(order+1)*(order+1);
	    
	    double x = mesh.globalCoords(0,vID,k);
	    double y = mesh.globalCoords(1,vID,k);
	    double z = mesh.globalCoords(2,vID,k);
	    
	    u(vID, iS, k) = a*std::exp(-std::pow(x-.5,2.0)-std::pow(y-.5,2.0)-std::pow(z-.5,2.0)/(2*sigmaSq));
	  }
	}
      }
    }
  }
  
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
  
  // explicit & diagonal => store nnz of A as off-diagonal vector
  darray diagA{nStages-1};
  diagA(0) = 0.5;
  diagA(1) = 0.5;
  diagA(2) = 1.0;
  
  darray uCurr{dofs, nStates, mesh.nElements};;
  darray ks{dofs, nStates, mesh.nElements, nStages};
  darray Dus{dofs, nStates, mesh.nElements, Mesh::DIM};
  
  std::cout << "Time stepping until tf = " << tf << std::endl;
  
  auto startTime = std::chrono::high_resolution_clock::now();
  
  // Loop over time steps
  for (int iStep = 0; iStep < timesteps; ++iStep) {
    std::cout << "time = " << iStep*dt << std::endl;
    
    // Use RK4 to compute k values at each stage
    for (int istage = 0; istage < nStages; ++istage) {
      
      // Updates uCurr = u+dt*a(s,s-1)*ks(:,s)
      rk4UpdateCurr(uCurr, diagA, ks, istage);
      
      // Updates ks(:,istage) based on DG method evaluated at uCurr
      rk4Rhs(uCurr, Dus, ks, istage);
      
    }
    
    // Use RK4 to move to next time step
    for (int istage = 0; istage < nStages; ++istage) {
      for (int iK = 0; iK < mesh.nElements; ++iK) {
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iN = 0; iN < dofs; ++iN) {
	    u(iN,iS,iK) += dt*b(istage)*ks(iN,iS,iK,istage);
	    if (iK == 0) {
	      std::cout << "u[" << iN << "] = " << u(iN,iS,iK) << "\n";
	    }
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
   Update current u variable based on diagonal Butcher tableau of RK4
*/
void Solver::rk4UpdateCurr(darray& uCurr, const darray& diagA, const darray& ks, int istage) const {
  
  // Update uCurr
  if (istage == 0) {
    for (int iK = 0; iK < mesh.nElements; ++iK) {
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iN = 0; iN < dofs; ++iN) {
	  uCurr(iN,iS,iK) = u(iN,iS,iK);
	}
      }
    }
  }
  else {
    for (int iK = 0; iK < mesh.nElements; ++iK) {
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iN = 0; iN < dofs; ++iN) {
	  uCurr(iN,iS,iK) = u(iN,iS,iK) + dt*diagA(istage-1)*ks(iN,iS,iK,istage-1);
	}
      }
    }
  }
  
}

/**
   Computes the RHS of Runge Kutta method for use with Local DG method.
   Updates ks(:,istage) with rhs evaluated at uCurr
*/
void Solver::rk4Rhs(const darray& uCurr, darray& Dus, darray& ks, int istage) const {
  
  // First solve for the Dus in each dimension according to:
  // Du_l = Mel\(-S_l*u + fluxesL(u))
  Dus.fill(0.0);
  localDGFlux(uCurr, Dus);
  for (int l = 0; l < Mesh::DIM; ++l) {
    
    // Du_l = -S_l*u + fluxesL(u)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
		dofs, nStates*mesh.nElements, dofs, -1.0, &Sels(0,0,l), dofs,
		uCurr.data(), dofs, 1.0, &Dus(0,0,0,l), dofs);
    
    // Du_l = Mel\Du_l
    MKL_INT ipiv[dofs];
    int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, dofs, nStates*mesh.nElements, 
			     Mel.data(), dofs, ipiv, &Dus(0,0,0,l), dofs);
    
    
  }
  
  // Now compute ks(:,istage) from uCurr and these Dus according to:
  // du/dt = Mel\( K*fc(u) + K*fv(u,Dus) - Fc(u) - Fv(u,Dus) )
  darray residual{&ks(0,0,0,istage), dofs, nStates, mesh.nElements};
  
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int iS = 0; iS < nStates; ++iS) {
      for (int iN = 0; iN < dofs; ++iN) {
	residual(iN,iS,iK,istage) = 0.0;
      }
    }
  }
  convectDGFlux(uCurr, residual);
  //viscousDGFlux(uCurr, Dus, residual); // TODO: write this
  
  // ks(:,istage) = -Fc(u)-Fv(u,Dus)
  cblas_dscal(dofs*nStates*mesh.nElements, -1.0, residual, 1);
  
  // ks(:,istage) += Kc*fc(u)
  convectDGVolume(uCurr, residual);
  
  // ks(:,istage) += Kv*fv(u)
  //viscousDGVolume(uCurr, Dus, residual); // TODO: uncomment this
  
  // ks(:,istage) = Mel\ks(:,istage)
  MKL_INT ipiv[dofs];
  int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, dofs, nStates*mesh.nElements, 
			   Mel.data(), dofs, ipiv, residual, dofs);
  
}


/**
   Local DG Flux: Computes Fl(u) for use in the local DG formulation of second-order terms.
   Uses a Lax-Friedrichs formulation for the flux term. TODO: Shouldn't we be using downwind?
   Updates globalFlux variable with added flux
*/
void Solver::localDGFlux(const darray& uCurr, darray& globalFlux) const {
  
  int nQ2D = Interp2D.size(0);
  
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
	
	for (int iS = 0; iS < nStates; ++iS) {
	  
	  // For every face, compute flux = integrate over face (fstar*phi_i)
	  // Must compute nStates of these flux integrals per face
	  darray integrand{Interp2D};
	  
	  for (int iFN = 0; iFN < mesh.nFNodes; ++iFN) {
	    auto uK = uCurr(mesh.efToN(iFN, iF), iS, iK);
	    auto uN = uCurr(mesh.efToN(iFN, nF), iS, nK);
	    
	    fStar(iFN, iF, iS, 1) = numericalFluxL(uK, uN, normalK, normalN);
	  }
	  
	  // Flux integrand = Interp2D*diag(fstar)
	  for (int iFN = 0; iFN < mesh.nFNodes; ++iFN) {
	    for (int iQ = 0; iQ < nQ2D; ++iQ) {
	      integrand(iQ, iFN) *= fStar(iFN, iF, iS, l);
	    }
	  }
	  
	  // contribution = integrand'*wQ2D
	  // TODO: this won't work as is because wQ2D is a vector
	  //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
	  //mesh.nFNodes, 1, nQ2D, 1.0, integrand.data(), nQ2D, 
	  //wQ2D.data(), nQ2D, 
	  //0.0, &faceContributions(0,iF,iS,l), mesh.nFNodes);
	  
	}
	
      }
    }
    
    // Add up face contributions into global flux array
    // Performed outside the above loops, thinking ahead to OpenMP...
    for (int l = 0; l < Mesh::DIM; ++l) {
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
	  for (int iFN = 0; iFN < mesh.nFNodes; ++iFN) {
	    globalFlux(mesh.efToN(iFN,iF), iS, iK, l) += faceContributions(iFN, iF, iS, l);
	  }
	}
      }
    }
    
  }
  
}

/** Evaluates the local DG flux function for this PDE at a given value uHere */
inline double Solver::numericalFluxL(double uK, double uN, double normalK, double normalN) const {
  
  // TODO: f(u) = u right now. This depends on PDE!
  auto fK = uK;
  auto fN = uN;
  // TODO: This appears to also be the Roe A value?
  double C = std::abs((fN-fK)/(uN-uK));
  
  return (fK+fN)/2.0 + (C/2.0)*(uN*normalN + uK*normalK);
  
}

/**
   Convect DG Flux: Computes Fc(u) for use in the Local DG formulation of the 
   1st-order convection term.
   Uses upwinding for the numerical flux. TODO: Use Per's Roe solver.
   Updates globalFlux variable with added flux
   TODO: Can you first evaluate fc(u) and then interpolate, or must you do fc(Interpolated u)?
*/
void Solver::convectDGFlux(const darray& uCurr, darray& globalFlux) const {
  
  int nFN = mesh.nFNodes;
  
  // Loop over all elements
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    
    // Initialize flux along faces
    darray fStar{nFN, nStates, Mesh::N_FACES}; // TODO: should we move this out of for loop
    darray faceContributions{nFN, nStates, Mesh::N_FACES};
    
    for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
      
      auto nF = mesh.eToF(iF, iK);
      auto nK = mesh.eToE(iF, iK);
      
      darray normalK{&mesh.normals(0, iF, iK), 3};
      darray normalN{&mesh.normals(0, nF, nK), 3};
      
      for (int iS = 0; iS < nStates; ++iS) {
	
	// For every face, compute flux = integrate over face (fstar*phi_i)
	// Must compute nStates of these flux integrals per face
	for (int iFN = 0; iFN < nFN; ++iFN) {
	  auto uK = uCurr(mesh.efToN(iFN, iF), iS, iK);
	  auto uN = uCurr(mesh.efToN(iFN, nF), iS, nK);
	  
	  fStar(iFN, iS, iF) = numericalFluxC(uK, uN, normalK, normalN);
	}
	
      }
      
      // Flux contribution = M2D*fstar
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
		  nFN, nStates, nFN, 1.0, &M2D(0,0,iF/2), nFN, 
		  &fStar(0,0,iF), nFN, 
		  0.0, &faceContributions(0,0,iF), nFN);
      
    }
    
    // Add up face contributions into global flux array
    // Performed outside the above loops, thinking ahead to OpenMP...
    for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iFN = 0; iFN < nFN; ++iFN) {
	  globalFlux(mesh.efToN(iFN,iF), iS, iK) += faceContributions(iFN, iS, iF);
	}
      }
    }
    
  }
  
}

/** Evaluates the convection DG flux function for this PDE at a given value uHere */
inline double Solver::numericalFluxC(double uK, double uN, const darray& normalK, const darray& normalN) const {
  
  double result = 0.0;
  
  // TODO: fc(u) = a*u right now. This depends on PDE!
  for (int l = 0; l < Mesh::DIM; ++l) {
    auto fK = fluxC(uK, l);
    auto fN = fluxC(uN, l);
    // upwinding assuming a is always positive
    result += (normalK(l) > 0.0 ? fK : fN)*normalK(l);
  }
  
  return result;
  
}

/** Evaluates the actual convection flux function for the PDE */
inline double Solver::fluxC(double uK, int l) const {
  return a[l]*uK;
}

/** Evaluates the volume integral term for convection in the RHS */
void Solver::convectDGVolume(const darray& uCurr, darray& residual) const {
  
  int nQ3D = Interp3D.size(0);
  
  // uInterp = Interp3D*uCurr
  darray uInterp{nQ3D, nStates, mesh.nElements};
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      nQ3D, nStates*mesh.nElements, dofs, 1.0, Interp3D.data(), dofs,
	      uCurr.data(), dofs, 0.0, uInterp.data(), nQ3D);
  
  // residual += Kels(:, l)*fc_l(uInterp)
  darray fc{nQ3D, nStates, mesh.nElements, Mesh::DIM};
  for (int l = 0; l < Mesh::DIM; ++l) {
    
    for (int k = 0; k < mesh.nElements; ++k) {
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iQ = 0; iQ < nQ3D; ++iQ) {
	  fc(iQ,iS,k,l) = fluxC(uInterp(iQ,iS,k), l);
	}
      }
    }
    
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		dofs, nStates*mesh.nElements, nQ3D, 1.0, &Kels(0,0,l), nQ3D,
		&fc(0,0,0,l), nQ3D, 1.0, residual.data(), dofs);
    
  }
  
}

/** Evaluates the actual viscosity flux function for the PDE */
inline double Solver::fluxV(double uK, const darray& DuK, int l) const {
  switch(l) {
  case 0:
    return 0.0;
    break;
  case 1:
    return 0.0;
    break;
  case 2:
    return 0.0;
    break;
  default:
    std::cerr << "How the hell...?" << std::endl;
  }
}

/** Evaluates the volume integral term for viscosity in the RHS */
void Solver::viscousDGVolume(const darray& uCurr, const darray& Dus, darray& residual) const {
  
  int nQ3D = Interp3D.size(0);
  
  // uInterp = Interp3D*uCurr
  darray uInterp{nQ3D, nStates, mesh.nElements};
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      nQ3D, nStates*mesh.nElements, dofs, 1.0, Interp3D.data(), dofs,
	      uCurr.data(), dofs, 0.0, uInterp.data(), nQ3D);
  // DuInterp = Interp3D*Dus
  darray DuInterp{nQ3D, nStates, mesh.nElements, Mesh::DIM};
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      nQ3D, nStates*mesh.nElements*Mesh::DIM, dofs, 1.0, Interp3D.data(), dofs,
	      Dus.data(), dofs, 0.0, DuInterp.data(), nQ3D);
  
  // residual += Kels(:, l)*fv_l(uInterp)
  darray fv{nQ3D, nStates, mesh.nElements, Mesh::DIM};
  for (int l = 0; l < Mesh::DIM; ++l) {
    
    for (int k = 0; k < mesh.nElements; ++k) {
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iQ = 0; iQ < nQ3D; ++iQ) {
	  darray DuHere{Mesh::DIM};
	  for (int l2 = 0; l2 < Mesh::DIM; ++l2) {
	    DuHere(l2) = DuInterp(iQ,iS,k,l2);
	  }
	  fv(iQ,iS,k,l) = fluxV(uInterp(iQ,iS,k), DuHere, l);
	}
      }
    }
    
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		dofs, nStates*mesh.nElements, nQ3D, 1.0, &Kels(0,0,l), nQ3D,
		&fv(0,0,0,l), nQ3D, 1.0, residual.data(), dofs);
    
  }
  
}
