#include "solver.h"

/** Default constructor */
Solver::Solver() : Solver{2, 10, 1.0, 1.0, Mesh{}} { }

/** Main constructor */
Solver::Solver(int _p, double dtSnap, double _tf, double _L, const Mesh& _mesh) :
  mesh{_mesh},
  mpi{_mesh.mpi},
  tf{_tf},
  dt{},
  timesteps{},
  stepsPerSnap{},
  order{_p},
  dofs{},
  nQV{},
  refNodes{},
  xQV{},
  nStates{},
  Mel{},
  Mipiv{},
  sToS{},
  Kels{},
  KelsF{},
  InterpF{},
  InterpV{},
  u{},
  p{}
{
  
  // Initialize nodes within elements
  chebyshev2D(order, refNodes);
  dofs = refNodes.size(1)*refNodes.size(2);
  mesh.setupNodes(refNodes, order);
  mpi.initDatatype(mesh.nFQNodes);
  mpi.initFaces(Mesh::N_FACES);
  
  // Initialize u and p
  nStates = Mesh::DIM*(Mesh::DIM+1)/2 + Mesh::DIM;
  // upper diagonal of E in Voigt notation, followed by v
  // u = [e11, e22, e12, v1, v2]
  u.realloc(dofs, nStates, mesh.nElements);
  initMaterialProps(); // sets mu, lambda, rho
  initialCondition(); // sets u
  initTimeStepping(dtSnap); // sets dt, timesteps, stepsPerSnap
  
  // Compute interpolation matrices
  precomputeInterpMatrices(); // sets nQV
  
  // Compute local matrices
  precomputeLocalMatrices();
  
  if (mpi.rank == mpi.ROOT) {
    std::cout << "dt = " << dt << std::endl;
    std::cout << "maxvel = " << p.C << std::endl;
  }
  
}

/** Precomputes all the interpolation matrices used by DG method */
void Solver::precomputeInterpMatrices() {
  
  // 1D interpolation matrix for use on faces
  darray chebyF;
  chebyshev1D(order, chebyF);
  darray xQF, wQF;
  gaussQuad1D(2*order, xQF, wQF);
  
  interpolationMatrix1D(chebyF, xQF, InterpF);
  
  // 2D interpolation matrix for use on elements
  darray chebyV;
  chebyshev2D(order, chebyV);
  darray wQV;
  nQV = gaussQuad2D(2*order, xQV, wQV);
  
  interpolationMatrix2D(chebyV, xQV, InterpV);
  
}

/** Precomputes all the local matrices used by DG method */
void Solver::precomputeLocalMatrices() {
  
  // Create nodal representation of the reference bases
  darray lV;
  legendre2D(order, refNodes, lV);
  
  darray coeffsPhi{order+1,order+1,order+1,order+1};
  for (int ipy = 0; ipy <= order; ++ipy) {
    for (int ipx = 0; ipx <= order; ++ipx) {
      coeffsPhi(ipx,ipy,ipx,ipy) = 1.0;
    }
  }
  lapack_int ipiv[dofs];
  LAPACKE_dgesv(LAPACK_COL_MAJOR, dofs, dofs, 
		lV.data(), dofs, ipiv, coeffsPhi.data(), dofs);
  
  // Compute reference bases on the quadrature points
  darray xQ, wQ;
  gaussQuad2D(2*order, xQ, wQ);
  darray polyQuad, dPolyQuad;
  legendre2D(order, xQ, polyQuad);
  dlegendre2D(order, xQ, dPolyQuad);
  
  darray phiQ{nQV,order+1,order+1};
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
	      nQV, dofs, dofs, 1.0, polyQuad.data(), nQV, 
	      coeffsPhi.data(), dofs, 0.0, phiQ.data(), nQV);
  darray dPhiQ{nQV,order+1,order+1,Mesh::DIM};
  for (int l = 0; l < Mesh::DIM; ++l) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
		nQV, dofs, dofs, 1.0, &dPolyQuad(0,0,0,l), nQV, 
		coeffsPhi.data(), dofs, 0.0, &dPhiQ(0,0,0,l), nQV);
  }
  
  // Store weights*phiQ in polyQuad to avoid recomputing every time
  for (int ipy = 0; ipy <= order; ++ipy) {
    for (int ipx = 0; ipx <= order; ++ipx) {
      for (int iQ = 0; iQ < nQV; ++iQ) {
	polyQuad(iQ, ipx,ipy) = phiQ(iQ, ipx,ipy)*wQ(iQ);
      }
    }
  }
  
  // Initialize mass matrices = integrate(ps*phi_i*phi_j)
  Mel.realloc(dofs,dofs,2,mesh.nElements);
  Mipiv.realloc(dofs,2,mesh.nElements);
  darray localJPI{nQV, order+1,order+1};
  darray rhoQ{nQV, mesh.nElements};
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      nQV, mesh.nElements, dofs, 1.0, InterpV.data(), nQV,
	      p.rho.data(), dofs, 0.0, rhoQ.data(), nQV);
  
  for (int k = 0; k < mesh.nElements; ++k) {
    
    // Compute Jacobian = det(Jacobian(Tk))
    double Jacobian = 1.0;
    for (int l = 0; l < Mesh::DIM; ++l) {
      Jacobian *= mesh.tempMapping(0,l,k);
    }
      
    for (int s = 0; s < 2; ++s) {
      
      // Compute localJPI = J*P*phiQ
      if (s == 0) {
	// P == I
	for (int ipy = 0; ipy <= order; ++ipy) {
	  for (int ipx = 0; ipx <= order; ++ipx) {
	    for (int iQ = 0; iQ < nQV; ++iQ) {
	      localJPI(iQ, ipx,ipy) = Jacobian*phiQ(iQ, ipx,ipy);
	    }
	  }
	}
      }
      else {
	// P == rho*I
	for (int ipy = 0; ipy <= order; ++ipy) {
	  for (int ipx = 0; ipx <= order; ++ipx) {
	    for (int iQ = 0; iQ < nQV; ++iQ) {
	      localJPI(iQ, ipx,ipy) = Jacobian*rhoQ(iQ)*phiQ(iQ, ipx,ipy);
	    }
	  }
	}
      }
      
      // Mel = Interp'*W*Jk*Ps*Interp
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
		  dofs, dofs, nQV, Jacobian, polyQuad.data(), nQV, 
		  localJPI.data(), nQV, 0.0, &Mel(0,0,s,k), dofs);
      
      // Mel overwritten with U*D*U'
      LAPACKE_dsytrf(LAPACK_COL_MAJOR, 'U', dofs,
		     &Mel(0,0,s,k), dofs, &Mipiv(0,s,k));
    }
  }
  
  // Set state to storage mapping
  sToS.realloc(nStates);
  // upper diagonal of strain tensor
  for (int s = 0; s < Mesh::DIM*(Mesh::DIM+1)/2; ++s) {
    sToS(s) = 0;
  }
  // velocities
  for (int s = Mesh::DIM*(Mesh::DIM+1)/2; s < nStates; ++s) {
    sToS(s) = 1;
  }
  
  // TODO: everything below here needs to work with new mappings
  darray alpha{Mesh::DIM};
  alpha(0) = mesh.minDX/2.0;
  alpha(1) = mesh.minDY/2.0;
  double Jacobian = alpha(0)*alpha(1);
  
  // Initialize K matrices = dx_l(phi_i)*weights
  Kels.realloc(nQV,dofs,Mesh::DIM);
  for (int l = 0; l < Mesh::DIM; ++l) {
    double scaleL = Jacobian/alpha(l);
    for (int iDofs = 0; iDofs < dofs; ++iDofs) {
      for (int iQ = 0; iQ < nQV; ++iQ) {
	Kels(iQ, iDofs, l) = dPhiQ(iQ, iDofs,0, l)*wQ(iQ)*scaleL;
      }
    }
  }
  
  // Initialize K matrix for use along faces
  darray xQF, wQF;
  int fSizeQ = gaussQuad1D(2*order, xQF, wQF);
  int fDofs = (order+1);
  KelsF.realloc(fSizeQ,fDofs, Mesh::DIM);
  for (int l = 0; l < Mesh::DIM; ++l) {
    double scaleL = Jacobian/alpha(l);
    for (int iDofs = 0; iDofs < fDofs; ++iDofs) {
      for (int iQ = 0; iQ < fSizeQ; ++iQ) {
	KelsF(iQ, iDofs, l) = InterpF(iQ,iDofs)*wQF(iQ)*scaleL;
      }
    }
  }
  
}

/** Initializes lambda, mu arrays according to file info */
void Solver::initMaterialProps() {
  
  darray vp, vs, rhoIn;
  darray deltas{Mesh::DIM};
  darray origins{Mesh::DIM};
  bool fileExists = readProps(vp, vs, rhoIn, origins, deltas);
  
  p.lambda.realloc(nQV, mesh.nElements);
  p.mu.realloc(nQV, mesh.nElements);
  p.rho.realloc(nQV, mesh.nElements);
  
  if (!fileExists) {
    // Initialize material properties to constants
    for (int k = 0; k < mesh.nElements; ++k) {
      for (int iQ = 0; iQ < nQV; ++iQ) {
	p.lambda(iQ,k) = p.rhoConst*(p.vpConst*p.vpConst - 2*p.vsConst*p.vsConst);
	p.mu(iQ,k) = p.rhoConst*(p.vsConst*p.vsConst);
	p.rho(iQ,k) = p.rhoConst;
      }
    }
  }
  else {
    // Interpolate data from grid onto quadrature points
    for (int k = 0; k < mesh.nElements; ++k) {
      for (int iQ = 0; iQ < nQV; ++iQ) {
	darray coord{Mesh::DIM};
	for (int l = 0; l < Mesh::DIM; ++l) {
	  coord(l) = mesh.tempMapping(0,l,k)*xQV(l,iQ)+mesh.tempMapping(1,l,k);
	}
	
	double vpi = gridInterp(coord, vp, origins, deltas);
	double vsi = gridInterp(coord, vs, origins, deltas);
	double rhoi = gridInterp(coord, rhoIn, origins, deltas);
	
	// Isotropic elastic media formula
	p.lambda(iQ,k) = rhoi*(vpi*vpi - 2*vsi*vsi);
	p.mu(iQ,k) = rhoi*(vsi*vsi);
	p.rho(iQ,k) = rhoi;
	
      }
    }
  }
  
}

/* Initialize time stepping information using CFL condition */
void Solver::initTimeStepping(double dtSnap) {
  
  // Compute p.C = maxvel = max(vp) throughout domain
  p.C = 0.0;
  for (int k = 0; k < mesh.nElements; ++k) {
    for (int iQ = 0; iQ < nQV; ++iQ) {
      double vpi = std::sqrt((p.lambda(iQ,k) + 2*p.mu(iQ,k))/p.rho(iQ,k));
      if (vpi > p.C)
	p.C = vpi;
    }
  }
  
  // Choose dt based on CFL condition
  dt = 0.1*std::min(mesh.minDX, mesh.minDY)/(p.C*(std::max(1, order*order)));
  // Ensure we will exactly end at tf
  timesteps = std::ceil(tf/dt);
  dt = tf/timesteps;
  // Compute number of time steps per dtSnap sec
  stepsPerSnap = std::ceil(dtSnap/dt);
  
}

/** Sets u according to an initial condition */
void Solver::initialCondition() {
  
  // Sin function allowing for periodic initial condition
  for (int k = 0; k < mesh.nElements; ++k) {
    for (int iy = 0; iy < order+1; ++iy) {
      for (int ix = 0; ix < order+1; ++ix) {
	int vID = ix + iy*(order+1);
	
	double x = mesh.globalCoords(0,vID,k);
	double y = mesh.globalCoords(1,vID,k);
	
	//u(vID, iS, k) = std::sin(2*x*M_PI)*std::sin(2*y*M_PI)*std::sin(2*z*M_PI);
	u(vID, 0, k) = -std::sin(x);
	u(vID, 1, k) = -std::sin(y);
	/*
	  u(vID, 0, k) = std::exp(-100*std::pow(x-.5, 2.0));
	  u(vID, 1, k) = std::exp(-100*std::pow(y-.5, 2.0));
	  u(vID, 2, k) = std::exp(-100*std::pow(z-.5, 2.0));
	*/
	
      }
    }
  }
  
}

/** Computes the true solution at time t in uTrue */
void Solver::trueSolution(darray& uTrue, double t) const {
  
  for (int k = 0; k < mesh.nElements; ++k) {
    for (int iy = 0; iy < order+1; ++iy) {
      for (int ix = 0; ix < order+1; ++ix) {
	int vID = ix + iy*(order+1);
	
	double x = mesh.globalCoords(0,vID,k);
	double y = mesh.globalCoords(1,vID,k);
	// True solution = convection u0(x-a*t)
	/*
	  uTrue(vID, iS, k) = std::sin(2*fmod(x-p.a[0]*t+5.0,1.0)*M_PI)
	  *std::sin(2*fmod(y-p.a[1]*t+5.0,1.0)*M_PI)
	  *std::sin(2*fmod(z-p.a[2]*t+5.0,1.0)*M_PI);
	  */
	
	// True solution = conv-diff
	/*uTrue(vID, 0, k) = -std::sin(x-p.a[0]*t)*std::exp(-p.eps*t);
	uTrue(vID, 1, k) = -std::sin(y-p.a[1]*t)*std::exp(-p.eps*t);
	*/
	
      }
    }
  }
}

/**
   Actually time step DG method according to RK4, 
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
  
  // Allocate working memory
  int nQF = mesh.nFQNodes;
  darray uCurr{dofs, nStates, mesh.nElements, 1};
  darray uInterpF{nQF, nStates, Mesh::N_FACES, mesh.nElements+mesh.nGElements, 1};
  darray uInterpV{nQV, nStates, mesh.nElements, 1};
  darray ks{dofs, nStates, mesh.nElements, nStages};
  darray uTrue{dofs, nStates, mesh.nElements, 1};
  
  int nBElems = max(mesh.mpiNBElems);
  darray toSend{nQF, nStates, nBElems, Mesh::DIM, MPIUtil::N_FACES};
  darray toRecv{nQF, nStates, nBElems, Mesh::DIM, MPIUtil::N_FACES};
  /**
     Requests for use in MPI sends/receives during rk4Rhs()
     2*face == send, 2*face+1 == recv
  */
  MPI_Request rk4Reqs[2*MPIUtil::N_FACES];
  
  if (mpi.rank == mpi.ROOT) {
    std::cout << "Time stepping until tf = " << tf << std::endl;
  }
  
  auto startTime = std::chrono::high_resolution_clock::now();
  
  // Loop over time steps
  for (int iStep = 0; iStep < timesteps; ++iStep) {
    
    /*if (mpi.rank == mpi.ROOT) {
      std::cout << "time = " << iStep*dt << std::endl;
    }*/
    
    if (iStep % stepsPerSnap == 0) {
      if (mpi.rank == mpi.ROOT) {
	auto endTime = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = endTime-startTime;
	std::cout << "Saving snapshot " << iStep/stepsPerSnap << "...\n";
	std::cout << "Elapsed time so far = " << elapsed.count() << std::endl;
      }
      trueSolution(uTrue, iStep*dt);
      // Output snapshot files
      bool success = initXYZVFile("output/xyzutrue.txt", Mesh::DIM, iStep/stepsPerSnap, "utrue", nStates);
      if (!success)
	exit(-1);
      success = exportToXYZVFile("output/xyzutrue.txt", iStep/stepsPerSnap, mesh.globalCoords, uTrue);
      if (!success)
	exit(-1);
      
      /* TODO: debugging for convection problem
      
      double norm = 0.0;
      for (int iK = 0; iK < mesh.nElements; ++iK) {
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iN = 0; iN < dofs; ++iN) {
	    uTrue(iN, iS, iK) -= u(iN, iS, iK);
	    if (std::abs(uTrue(iN, iS, iK)) > norm) {
	      norm = std::abs(uTrue(iN, iS, iK));
	    }
	  }
	}
      }
      std::cout << "infinity norm at time " << iStep*dt << " = " << norm << std::endl;
      // END TODO */
      
    }
    
    // Use RK4 to compute k values at each stage
    for (int istage = 0; istage < nStages; ++istage) {
      
      // Updates uCurr = u+dt*a(s,s-1)*ks(:,s)
      rk4UpdateCurr(uCurr, diagA, ks, istage);
      
      // Updates ks(:,istage) = rhs(uCurr) based on DG method
      rk4Rhs(uCurr, uInterpF, uInterpV, 
	     toSend, toRecv, rk4Reqs, ks, istage);
      
    }
    
    // Use RK4 to move to next time step
    for (int istage = 0; istage < nStages; ++istage) {
      for (int iK = 0; iK < mesh.nElements; ++iK) {
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iN = 0; iN < dofs; ++iN) {
	    u(iN,iS,iK) += dt*b(istage)*ks(iN,iS,iK,istage);
	    
	    if (istage == nStages-1 && iK == 0 && iS == 5 && mpi.rank == mpi.ROOT) {
	      std::cout << "u[" << iN << "] = " << u(iN,iS,iK) << std::endl;
	    }
	  }
	}
      }
    }
    
  }
  
  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = endTime-startTime;
  if (mpi.rank == mpi.ROOT) {
    std::cout << "Finished time stepping. Time elapsed = " << elapsed.count() << std::endl;
  }
  
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
   Computes the RHS of Runge Kutta method for use with DG method.
   Updates ks(:,istage) with rhs evaluated at uCurr
*/
void Solver::rk4Rhs(const darray& uCurr, darray& uInterpF, darray& uInterpV, 
		    darray& toSend, darray& toRecv, MPI_Request * rk4Reqs, darray& ks, int istage) const {
  
  // Interpolate uCurr once
  interpolate(uCurr, uInterpF, uInterpV, toSend, toRecv, rk4Reqs, 1);
  
  // Now compute ks(:,istage) from uInterp according to:
  // ks(:,istage) = Mel\( K*fc(u) - Fc(u) )
  
  darray residual{&ks(0,0,0,istage), dofs, nStates, mesh.nElements};
  residual.fill(0.0);
  
  // ks(:,istage) = -Fc(u)
  convectDGFlux(uInterpF, residual);
  cblas_dscal(dofs*nStates*mesh.nElements, -1.0, residual.data(), 1);
  
  // ks(:,istage) += Kc*fc(u)
  convectDGVolume(uInterpV, residual);
  
  // ks(:,istage) = Mel\ks(:,istage)
  for (int k = 0; k < mesh.nElements; ++k) {
    for (int s = 0; s < nStates; ++s) {
      LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'U', dofs, 1,
		     &Mel(0,0,s,k), dofs, &Mipiv(0,s,k), &residual(0,s,k), dofs);
    }
  }
  
  
}

/**
   Interpolates u on faces to 2D quadrature points and stores in uInterpF.
   Interpolates u onto 3D quadrature points and stores in uInterpV.
*/
void Solver::interpolate(const darray& curr, darray& toInterpF, darray& toInterpV,
			 darray& toSend, darray& toRecv, MPI_Request * rk4Reqs, int dim) const {
  
  // First grab u on faces and pack into array uOnFaces
  int nFN = mesh.nFNodes;
  int nQF = mesh.nFQNodes;
  darray onFaces{nFN, nStates, Mesh::N_FACES, mesh.nElements, dim};
  for (int l = 0; l < dim; ++l) {
    for (int iK = 0; iK < mesh.nElements; ++iK) {
      for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iFN = 0; iFN < nFN; ++iFN) {
	    onFaces(iFN, iS, iF, iK, l) = curr(mesh.efToN(iFN, iF), iS, iK, l);
	  }
	}
      }
    }
    
    // Face interpolation toInterpF = InterpF*uOnFaces
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		nQF, nStates*Mesh::N_FACES*mesh.nElements, nFN, 1.0, 
		InterpF.data(), nQF, &onFaces(0,0,0,0,l), nFN, 
		0.0, &toInterpF(0,0,0,0,l), nQF);
  }
  
  mpiStartComm(toInterpF, dim, toSend, toRecv, rk4Reqs);
  
  // Volume interpolation toInterpV = InterpV*curr
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      nQV, nStates*mesh.nElements*dim, dofs, 1.0, InterpV.data(), nQV,
	      curr.data(), dofs, 0.0, toInterpV.data(), nQV);
  
  // TODO: move after all interior elements have been computed on
  mpiEndComm(toInterpF, dim, toRecv, rk4Reqs);
  
}


/**
   Convect DG Flux: Computes Fc(u) for use in the DG formulation of the 
   1st-order convection term.
   Uses Lax-Friedrichs for the numerical flux. 
   Updates residual variable with added flux.
*/
void Solver::convectDGFlux(const darray& uInterpF, darray& residual) const {
  
  int nFN = mesh.nFNodes;
  int nQF = mesh.nFQNodes;
  
  // Initialize flux along faces
  darray fStar{nQF, nStates};
  darray faceContributions{nFN, nStates};
  
  darray fluxes{nStates};
  darray uK{nStates};
  darray uN{nStates};
  
  // Loop over all elements
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
      
      // For every face, compute fstar = fc*(InterpF*u)
      auto nF = mesh.eToF(iF, iK);
      auto nK = mesh.eToE(iF, iK);
      
      darray normalK{&mesh.normals(0, iF, iK), Mesh::DIM};
      
      for (int iFQ = 0; iFQ < nQF; ++iFQ) {
	
	// Get all state variables at this point
	for (int iS = 0; iS < nStates; ++iS) {
	  uK(iS) = uInterpF(iFQ, iS, iF, iK);
	  uN(iS) = uInterpF(iFQ, iS, nF, nK);
	}
	// Get material properties at this point
	double lambdaK = p.lambda(mesh.efToQ(iFQ, iF), iK);
	double lambdaN = p.lambda(mesh.efToQ(iFQ, nF), nK);
	double muK = p.lambda(mesh.efToQ(iFQ, iF), iK);
	double muN = p.lambda(mesh.efToQ(iFQ, nF), nK);
	double rhoK = p.rho(mesh.efToQ(iFQ, iF), iK);
	double rhoN = p.rho(mesh.efToQ(iFQ, nF), nK);
	
	// Compute fluxes = F*(u+,u-)'*n
	numericalFluxC(uN, uK, normalK, fluxes,
		       lambdaN, muN, rhoN, lambdaK, muK, rhoK);
	// Copy flux data into fStar
	for (int iS = 0; iS < nStates; ++iS) {
	  fStar(iFQ, iS) = fluxes(iS);
	}
	
      }
      
      // Flux contribution = KelsF(:,l)'*fstar
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
		  nFN, nStates, nQF, 1.0, &KelsF(0,0,iF/2), nQF, 
		  fStar.data(), nQF, 
		  0.0, faceContributions.data(), nFN);
      
      // Add up face contributions into global residual array
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iFN = 0; iFN < nFN; ++iFN) {
	  residual(mesh.efToN(iFN,iF), iS, iK) += faceContributions(iFN, iS);
	}
      }
      
    }
  }
  
}

/** Evaluates the volume integral term for convection in the RHS */
void Solver::convectDGVolume(const darray& uInterpV, darray& residual) const {
  
  // Contains all flux information
  darray fc{nQV, nStates, mesh.nElements, Mesh::DIM};
  
  // Temporary arrays for computing fluxes
  darray fluxes{nStates, Mesh::DIM};
  darray uK{nStates};
  
  // Loop over all elements, points
  for (int k = 0; k < mesh.nElements; ++k) {
    for (int iQ = 0; iQ < nQV; ++iQ) {
      
      // Copy data into uK
      for (int iS = 0; iS < nStates; ++iS) {
	uK(iS) = uInterpV(iQ,iS,k);
      }
      // Compute fluxes
      fluxC(uK, fluxes, p.lambda(iQ,k), p.mu(iQ,k));
      // Copy data into fc
      for (int l = 0; l < Mesh::DIM; ++l) {
	for (int iS = 0; iS < nStates; ++iS) {
	  fc(iQ,iS,k,l) = fluxes(iS, l);
	}
      }
      
    }
  }
  
  // TODO: is it faster to do matrix vector multiplication instead of having to reorder data into huge fc array?
  
  // residual += Kels(:, l)*fc_l(uInterpV)
  for (int l = 0; l < Mesh::DIM; ++l) {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		dofs, nStates*mesh.nElements, nQV, 1.0, &Kels(0,0,l), nQV,
		&fc(0,0,0,l), nQV, 1.0, residual.data(), dofs);
  }
  
}


/**
   MPI communication: Start nonblocking Sends/Recvs of pre-interpolated data to other tasks.
   First packs data from interpolated face data into send arrays.
   Assumes interpolated is of size (nQF, nStates, Mesh::N_FACES, nElements+nGElements, dim)
*/
void Solver::mpiStartComm(const darray& interpolated, int dim, darray& toSend, darray& toRecv, MPI_Request * rk4Reqs) const {
  
  int nQF = mesh.nFQNodes;
  
  // Pack face data to send
  for (int iF = 0; iF < MPIUtil::N_FACES; ++iF) {
    
    for (int l = 0; l < dim; ++l) {
      for (int bK = 0; bK < mesh.mpiNBElems(iF); ++bK) {
	int iK = mesh.mpibeToE(bK, iF);
	
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iQ = 0; iQ < nQF; ++iQ) {
	    toSend(iQ, iS, bK, l, iF) = interpolated(iQ, iS, mpi.faceMap(iF), iK, l);
	  }
	}
      }
    }
    
  }
  
  // Actually send/recv the data
  for (int iF = 0; iF < MPIUtil::N_FACES; ++iF) {
    
    MPI_Isend(&toSend(0,0,0,0,iF), nStates*mesh.mpiNBElems(iF)*dim, 
	      mpi.MPI_FACE, mpi.neighbors(iF), iF,
	      mpi.cartComm, &rk4Reqs[2*iF]);
    
    MPI_Irecv(&toRecv(0,0,0,0,iF), nStates*mesh.mpiNBElems(iF)*dim,
	      mpi.MPI_FACE, mpi.neighbors(iF), mpi.tags(iF), 
	      mpi.cartComm, &rk4Reqs[2*iF+1]);
    
  }
  
}

/**
   MPI communication: Finalizes nonblocking Sends/Recvs of uInterpF.
   Then unpacks data from recv arrays into interpolated.
   Assumes interpolated is of size (nQF, nStates, Mesh::N_FACES, nElements, dim)
*/
void Solver::mpiEndComm(darray& interpolated, int dim, const darray& toRecv, MPI_Request * rk4Reqs) const {
  
  int nQF = mesh.nFQNodes;
  
  // Finalizes sends/recvs
  // TODO: would there really be any benefit to waiting on only some of the faces first?
  MPI_Waitall(2*MPIUtil::N_FACES, rk4Reqs, MPI_STATUSES_IGNORE);
  
  // Unpack data
  for (int iF = 0; iF < MPIUtil::N_FACES; ++iF) {
    
    for (int l = 0; l < dim; ++l) {
      for (int bK = 0; bK < mesh.mpiNBElems(iF); ++bK) {
	auto iK = mesh.mpibeToE(bK, iF);
	auto nF = mesh.eToF(mpi.faceMap(iF), iK);
	auto nK = mesh.eToE(mpi.faceMap(iF), iK);
	
	
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iQ = 0; iQ < nQF; ++iQ) {
	    interpolated(iQ, iS, nF, nK, l) = toRecv(iQ, iS, bK, l, iF);
	  }
	}
      }
    }
    
  }
  
}





//////////////////////////////////////////////////////////////////////////////////


/** Evaluates the actual convection flux function for the PDE */
inline void Solver::fluxC(const darray& uK, darray& fluxes, double lambda, double mu) const {
  
  // Elastic wave equation flux term
  int nstrains = Mesh::DIM*(Mesh::DIM+1)/2;
  fluxes.fill(0.0);
  
  // -(vxI + Ixv)/2.0
  fluxes(0,0) = -uK(nstrains+0);
  fluxes(0,1) = 0.0;
  fluxes(1,0) = 0.0;
  fluxes(1,1) = -uK(nstrains+1);
  fluxes(2,0) = -uK(nstrains+1)/2.0;
  fluxes(2,1) = -uK(nstrains+0)/2.0;
  
  // -S = -CE
  fluxes(nstrains+0,0) = -(lambda+2*mu)*uK(0) - lambda*uK(1);
  fluxes(nstrains+1,1) = -(lambda+2*mu)*uK(1) - lambda*uK(0);
  fluxes(nstrains+0,1) = -(2*mu)*uK(2);
  fluxes(nstrains+1,0) = -(2*mu)*uK(2);
  
  /*
  // Convection-diffusion along each dimension
  fluxes.fill(0.0);
  for (int l = 0; l < Mesh::DIM; ++l) {
    fluxes(l,l) = p.a[l]*uK(l);
  }
  */
  
  /*
  // Fi2d for Navier-Stokes
  double t1;
  double t3;
  double t5;
  double t6;
  double t7;
  double t8;
  double t11;
  double t14;
  fluxes[0] = uK[1];
  t1 = std::pow(uK[1], 2.0);
  t3 = 1.0 / uK[0];
  t5 = uK[3];
  t6 = 2.0 / 5.0 * t5;
  t7 = uK[2];
  t8 = t7 * t7;
  t11 = (t1 + t8) * t3 / 5.0;
  fluxes[1] = t1 * t3 + t6 - t11;
  fluxes[2] = fluxes[0] * t7 * t3;
  t14 = 7.0/5.0 * t5 - t11;
  fluxes[3] = fluxes[0] * t14 * t3;
  fluxes[4] = t7;
  fluxes[5] = fluxes[2];
  fluxes[6] = t8 * t3 + t6 - t11;
  fluxes[7] = fluxes[4] * t14 * t3;
  */
}


/**
   Evaluates the convection numerical flux function for this PDE for all states, 
   dotted with the normal, storing output in fluxes.
*/
void Solver::numericalFluxC(const darray& uN, const darray& uK, 
			    const darray& normalK, darray& fluxes,
			    double lambdaN, double muN, double rhoN, 
			    double lambdaK, double muK, double rhoK) const {
  
  // Lax-Friedrichs flux for elastic wave equation
  darray FK{nStates, Mesh::DIM};
  darray FN{nStates, Mesh::DIM};
  
  fluxC(uK, FK, lambdaK, muK);
  fluxC(uN, FN, lambdaN, muN);
  
  // maximum vel = vp = avg of vp at each quadrature point
  double vpK = std::sqrt(lambdaK + 2*muK)/rhoK;
  double vpN = std::sqrt(lambdaN + 2*muN)/rhoN;
  double C = (vpK+vpN)/2.0;
  //double C = std::max(vpK, vpN); // TODO: is this better?
  
  fluxes.fill(0.0);
  for (int iS = 0; iS < nStates; ++iS) {
    for (int l = 0; l < Mesh::DIM; ++l) {
      fluxes(iS) += (FN(iS,l) + FK(iS,l))/2.0*normalK(l)
	- C/2.0*(uN(iS) - uK(iS))*normalK(l)*normalK(l);
      // TODO: is this negative sign correct?
    }
  }
  
  /*
  // Upwinding for convection-diffusion in 3 dimensions
  darray FK{nStates,Mesh::DIM};
  darray FN{nStates,Mesh::DIM};
  
  fluxC(uK, FK);
  fluxC(uN, FN);
  
  fluxes.fill(0.0);
  for (int l = 0; l < Mesh::DIM; ++l) {
    for (int iS = 0; iS < nStates; ++iS) {
      // upwinding assuming a[l] is always positive
      fluxes(iS) += (normalK(l) > 0.0 ? FK(iS, l) : FN(iS, l))*normalK(l);
    }
  }
  */
  
  /*
  // roe2d from Navier-Stokes
  double t17;
  double t46;
  double t155;
  double t23;
  double t59;
  double t35;
  double t76;
  double t47;
  double t79;
  double t49;
  double t50;
  double t51;
  double t80;
  double t2;
  double t13;
  double t81;
  double t82;
  double t22;
  double t98;
  double t99;
  double t8;
  double t85;
  double t52;
  double t55;
  double t58;
  double t37;
  double t38;
  double t39;
  double t16;
  double t69;
  double t100;
  double t90;
  double t106;
  double t105;
  double t103;
  double t102;
  double t109;
  double t108;
  double t3;
  double t5;
  double t60;
  double t61;
  double t112;
  double t111;
  double t113;
  double t94;
  double t119;
  double t122;
  double t18;
  double t19;
  double t125;
  double t41;
  double t30;
  double t97;
  double t128;
  double t130;
  double t26;
  double t64;
  double t10;
  double t134;
  double t1;
  double t137;
  double t136;
  double t143;
  double t148;
  t1 = uN[0];
  t2 = uN[1];
  t3 = 0.1e1 / t1;
  t5 = normalK[0];
  t8 = uN[2];
  t10 = normalK[1];
  t13 = 0.10e1 * t2 * t3 * t5 + 0.10e1 * t8 * t3 * t10;
  t16 = uK[0];
  t17 = uK[1];
  t18 = 0.1e1 / t16;
  t19 = t17 * t18;
  t22 = uK[2];
  t23 = t22 * t18;
  t26 = 0.10e1 * t19 * t5 + 0.10e1 * t23 * t10;
  t30 = std::sqrt(t1 * t18);
  t35 = 0.100e1 * t30 * t2 * t3 + 0.10e1 * t19;
  t37 = 0.10e1 * t30 + 0.10e1;
  t38 = 0.1e1 / t37;
  t39 = t35 * t38;
  t41 = 0.10e1 * t39 * t5;
  t46 = 0.100e1 * t30 * t8 * t3 + 0.10e1 * t23;
  t47 = t46 * t38;
  t49 = 0.10e1 * t47 * t10;
  t50 = t41 + t49;
  t51 = std::abs(t50);
  t52 = t1 - t16;
  t55 = uN[3];
  t58 = 0.4e0 * t55;
  t59 = t2 * t2;
  t60 = t1 * t1;
  t61 = 0.1e1 / t60;
  t64 = t8 * t8;
  t69 = 0.20e0 * t1 * (0.100e1 * t59 * t61 + 0.100e1 * t64 * t61);
  t76 = uK[3];
  t79 = 0.4e0 * t76;
  t80 = t17 * t17;
  t81 = t16 * t16;
  t82 = 0.1e1 / t81;
  t85 = t22 * t22;
  t90 = 0.20e0 * t16 * (0.100e1 * t80 * t82 + 0.100e1 * t85 * t82);
  t94 = 0.10e1 * t30 * (0.10e1 * t55 * t3 + 0.10e1 * (t58 - t69) * t3) + 0.10e1 * t76 * t18 + 0.10e1 * (t79 - t90) * t18;
  t97 = t35 * t35;
  t98 = t37 * t37;
  t99 = 0.1e1 / t98;
  t100 = t97 * t99;
  t102 = t46 * t46;
  t103 = t102 * t99;
  t105 = 0.40e0 * t94 * t38 - 0.2000e0 * t100 - 0.2000e0 * t103;
  t106 = std::sqrt(t105);
  t108 = std::abs(t41 + t49 + t106);
  t109 = 0.5e0 * t108;
  t111 = std::abs(-t41 - t49 + t106);
  t112 = 0.5e0 * t111;
  t113 = t109 + t112 - t51;
  t119 = t2 - t17;
  t122 = t8 - t22;
  t125 = 0.4e0 * (0.500e0 * t100 + 0.500e0 * t103) * t52 - 0.40e0 * t39 * t119 - 0.40e0 * t47 * t122 + t58 - t79;
  t128 = t113 * t125 / t105;
  t130 = t109 - t112;
  t134 = -t50 * t52 + t119 * t5 + t122 * t10;
  t136 = 0.1e1 / t106;
  t137 = t130 * t134 * t136;
  fluxes[0] = 0.5e0 * t1 * t13 + 0.5e0 * t16 * t26 - 0.5e0 * t51 * t52 - 0.5e0 * t128 - 0.5e0 * t137;
  t143 = t58 - t69 + t79 - t90;
  t148 = t128 + t137;
  t155 = t130 * t125 * t136 + t113 * t134;
  fluxes[1] = 0.5e0 * t2 * t13 + 0.5e0 * t17 * t26 + 0.5e0 * t5 * t143 - 0.5e0 * t51 * t119 - 0.50e0 * t148 * t35 * t38 - 0.5e0 * t155 * t5;
  fluxes[2] = 0.5e0 * t8 * t13 + 0.5e0 * t22 * t26 + 0.5e0 * t10 * t143 - 0.5e0 * t51 * t122 - 0.50e0 * t148 * t46 * t38 - 0.5e0 * t155 * t10;
  fluxes[3] = 0.5e0 * (0.14e1 * t55 - t69) * t13 + 0.5e0 * (0.14e1 * t76 - t90) * t26 - 0.5e0 * t51 * (t55 - t76) - 0.50e0 * t148 * t94 * t38 - 0.5e0 * t155 * t50;
  */
}
