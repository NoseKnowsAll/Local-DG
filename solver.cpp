#include "solver.h"

/** Default constructor */
Solver::Solver() : Solver{2, 10, 1.0, 1.0, Mesh{}} { }

/** Main constructor */
Solver::Solver(int _p, int _dtSnaps, double _tf, double _L, const Mesh& _mesh) :
  mesh{_mesh},
  mpi{_mesh.mpi},
  tf{_tf},
  dt{},
  timesteps{},
  dtSnaps{_dtSnaps},
  order{_p},
  dofs{},
  refNodes{},
  nStates{},
  Mel{},
  Mipiv{},
  Sels{},
  Kels{},
  KelsF{},
  InterpF{},
  InterpV{},
  u{},
  p{}
{
  
  // Initialize time stepping and physics information
  //double maxVel = std::accumulate(p.a, p.a+Mesh::DIM, 0.0);
  p.pars[0]  = 1600.0; // Reynolds
  p.pars[1]  = .71;    // Prandtl
  p.gamma    = 1.4;    // adiabatic gas constant
  p.M0       = 0.1;    // Mach
  p.R        = 8.314;  // ideal gas constant
  p.rho0     = 1.0;                    // Necessary
  p.V0       = 1.0;                    // Necessary
  p.L        = _L;
  
  //p.V0 = p.params[0]*p.mu/(p.rho0*p.L);
  p.c0 = p.V0/p.M0;
  p.p0 = p.rho0*p.c0*p.c0/p.gamma;      // Necessary
  p.tc = p.L/p.V0;                      // Necessary
  p.T0 = p.p0/(p.R*p.rho0);
  double maxVel = p.V0;
  
  // Choose dt based on CFL condition - DEPENDS ON PDE
  dt = 0.01*std::min(mesh.minDX, mesh.minDY)/(maxVel*(2*order+1));
  // Change tf = 10*tc in order to visualize most turbulent structures
  //tf = 10*p.tc;
  // Ensure we will exactly end at tf
  timesteps = std::ceil(tf/dt);
  dt = tf/timesteps;
  
  // Initialize nodes within elements
  chebyshev2D(order, refNodes);
  dofs = refNodes.size(1)*refNodes.size(2);
  mesh.setupNodes(refNodes, order);
  mpi.initDatatype(mesh.nFQNodes);
  mpi.initFaces(Mesh::N_FACES);
  
  nStates = 4;
  u.realloc(dofs, nStates, mesh.nElements);
  initialCondition(); // sets u

  std::cout << "solver1" << std::endl;
  
  // Compute interpolation matrices
  precomputeInterpMatrices();

  std::cout << "solver2" << std::endl;
  
  // Compute local matrices
  precomputeLocalMatrices();

  std::cout << "solver3" << std::endl;
  
}

/** Precomputes all the interpolation matrices used by Local DG method */
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
  darray xQV, wQV;
  gaussQuad2D(2*order, xQV, wQV);
  
  interpolationMatrix2D(chebyV, xQV, InterpV);
  
}

/** Precomputes all the local matrices used by Local DG method */
void Solver::precomputeLocalMatrices() {
  
  // Create nodal representation of the reference bases
  darray lV;
  legendre2D(order, refNodes, lV);

  std::cout << "matrix0" << std::endl;
  
  darray coeffsPhi{order+1,order+1,order+1,order+1};
  for (int ipy = 0; ipy <= order; ++ipy) {
    for (int ipx = 0; ipx <= order; ++ipx) {
      coeffsPhi(ipx,ipy,ipx,ipy) = 1.0;
    }
  }
  lapack_int ipiv[dofs];
  LAPACKE_dgesv(LAPACK_COL_MAJOR, dofs, dofs, 
		lV.data(), dofs, ipiv, coeffsPhi.data(), dofs);
  
  std::cout << "matrix1" << std::endl;
  
  // Compute reference bases on the quadrature points
  darray xQ, wQ;
  int sizeQ = gaussQuad2D(2*order, xQ, wQ);
  darray polyQuad, dPolyQuad;
  legendre2D(order, xQ, polyQuad);
  dlegendre2D(order, xQ, dPolyQuad);

  std::cout << "matrix2" << std::endl;
  
  darray phiQ{sizeQ,order+1,order+1};
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
	      sizeQ, dofs, dofs, 1.0, polyQuad.data(), sizeQ, 
	      coeffsPhi.data(), dofs, 0.0, phiQ.data(), sizeQ);
  
  std::cout << "matrix2.5" << std::endl;
  
  darray dPhiQ{sizeQ,order+1,order+1,Mesh::DIM};
  for (int l = 0; l < Mesh::DIM; ++l) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
		sizeQ, dofs, dofs, 1.0, &dPolyQuad(0,0,0,l), sizeQ, 
		coeffsPhi.data(), dofs, 0.0, &dPhiQ(0,0,0,l), sizeQ);
  }

  std::cout << "matrix3" << std::endl;
  
  // Store weights*phiQ in polyQuad to avoid recomputing 4 times
  for (int ipy = 0; ipy <= order; ++ipy) {
    for (int ipx = 0; ipx <= order; ++ipx) {
      for (int iQ = 0; iQ < sizeQ; ++iQ) {
	polyQuad(iQ, ipx,ipy) = phiQ(iQ, ipx,ipy)*wQ(iQ);
      }
    }
  }
  
  // Initialize mass matrix = integrate(phi_i*phi_j)
  // assumes all elements are the same size...
  darray alpha{Mesh::DIM};
  alpha(0) = mesh.minDX/2.0;
  alpha(1) = mesh.minDY/2.0;
  double Jacobian = alpha(0)*alpha(1);
  Mel.realloc(dofs,dofs);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
	      dofs, dofs, sizeQ, Jacobian, polyQuad.data(), sizeQ, 
	      phiQ.data(), sizeQ, 0.0, Mel.data(), dofs);
  Mipiv.realloc(dofs);
  // Mel overwritten with U*D*U'
  LAPACKE_dsytrf(LAPACK_COL_MAJOR, 'U', dofs,
		 Mel.data(), dofs, Mipiv.data());

  std::cout << "matrix4" << std::endl;
  
  // Initialize stiffness matrices = integrate(dx_l(phi_i)*phi_j)
  Sels.realloc(dofs,dofs,Mesh::DIM);
  for (int l = 0; l < Mesh::DIM; ++l) {
    double scaleL = Jacobian/alpha(l);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
		dofs, dofs, sizeQ, scaleL, &dPhiQ(0,0,0,l), sizeQ, 
		polyQuad.data(), sizeQ, 0.0, &Sels(0,0,l), dofs);
  }
  
  // Initialize K matrices = dx_l(phi_i)*weights
  Kels.realloc(sizeQ,dofs,Mesh::DIM);
  for (int l = 0; l < Mesh::DIM; ++l) {
    double scaleL = Jacobian/alpha(l);
    for (int iDofs = 0; iDofs < dofs; ++iDofs) {
      for (int iQ = 0; iQ < sizeQ; ++iQ) {
	Kels(iQ, iDofs, l) = dPhiQ(iQ, iDofs,0, l)*wQ(iQ)*scaleL;
      }
    }
  }

  std::cout << "matrix5" << std::endl;
  
  // Initialize mass matrix for use along faces
  darray xQF, wQF;
  int fSizeQ = gaussQuad1D(2*order, xQF, wQF);
  int fDofs = (order+1);
  KelsF.realloc(fSizeQ,fDofs, Mesh::DIM);

  std::cout << "matrix5.5" << std::endl;
  for (int l = 0; l < Mesh::DIM; ++l) {
    double scaleL = Jacobian/alpha(l);
    for (int iDofs = 0; iDofs < fDofs; ++iDofs) {
      for (int iQ = 0; iQ < fSizeQ; ++iQ) {
	KelsF(iQ, iDofs, l) = InterpF(iQ,iDofs)*wQF(iQ)*scaleL;
      }
    }
  }

  std::cout << "matrix6" << std::endl;
  
}

void Solver::initialCondition() {
  
  // Sin function allowing for periodic initial condition
  for (int k = 0; k < mesh.nElements; ++k) {
    for (int iy = 0; iy < order+1; ++iy) {
      for (int ix = 0; ix < order+1; ++ix) {
	int vID = ix + iy*(order+1);
	
	double x = mesh.globalCoords(0,vID,k);
	double y = mesh.globalCoords(1,vID,k);
	
	//u(vID, iS, k) = std::sin(2*x*M_PI)*std::sin(2*y*M_PI)*std::sin(2*z*M_PI);
	/*
	  u(vID, 0, k) = std::exp(-100*std::pow(x-.5, 2.0));
	  u(vID, 1, k) = std::exp(-100*std::pow(y-.5, 2.0));
	  u(vID, 2, k) = std::exp(-100*std::pow(z-.5, 2.0));
	*/
	
	// p
	double pressure = p.p0 + (p.rho0*p.V0*p.V0/16.0)
	  *(std::cos(2*x/p.L)+std::cos(2*y/p.L));
	// rho
	u(vID, 0, k) = pressure/(p.R*p.T0);
	// rho*v_0
	u(vID, 1, k) = p.rho0*p.V0*std::sin(x/p.L)*std::cos(y/p.L);
	// rho*v_1
	u(vID, 2, k) = -p.rho0*p.V0*std::cos(x/p.L)*std::sin(y/p.L);
	// rho*E
	u(vID, 3, k) = pressure/(p.gamma-1) 
	    + ( u(vID,1,k)*u(vID,1,k) + u(vID,2,k)*u(vID,2,k) )/(2.0*u(vID,0,k));
	
      }
    }
  }
  
}

/** Computes the true convection solution at time t for the convection problem */
void Solver::trueSolution(darray& uTrue, double t) const {
  
  //int N = 2;
  
  for (int k = 0; k < mesh.nElements; ++k) {
    for (int iy = 0; iy < order+1; ++iy) {
      for (int ix = 0; ix < order+1; ++ix) {
	int vID = ix + iy*(order+1);
	
	double x = mesh.globalCoords(0,vID,k);
	double y = mesh.globalCoords(1,vID,k);
	// True solution = initial solution u0(x-a*t)
	/*
	  uTrue(vID, iS, k) = std::sin(2*fmod(x-p.a[0]*t+5.0,1.0)*M_PI)
	  *std::sin(2*fmod(y-p.a[1]*t+5.0,1.0)*M_PI)
	  *std::sin(2*fmod(z-p.a[2]*t+5.0,1.0)*M_PI);
	  */
	/*
	// True solution = conv-diff
	uTrue(vID, 0, k) = 0.0;
	for (int i = -N; i <= N; ++i) {
	  uTrue(vID, 0, k) += std::exp(-100/(1+400*p.eps*t)*
	       (std::pow(std::fmod(x-t,1.0)-.5+i,2.0)))
	    /std::sqrt(1+400*p.eps*t);
	}
	uTrue(vID, 1, k) = 0.0;
	for (int i = -N; i <= N; ++i) {
	  uTrue(vID, 1, k) += std::exp(-100/(1+400*p.eps*t)*
	       (std::pow(std::fmod(y-t,1.0)-.5+i,2.0)))
	    /std::sqrt(1+400*p.eps*t);
	}
	uTrue(vID, 2, k) = 0.0;
	for (int i = -N; i <= N; ++i) {
	  uTrue(vID, 2, k) += std::exp(-100/(1+400*p.eps*t)*
	       (std::pow(std::fmod(z-t,1.0)-.5+i,2.0)))
	    /std::sqrt(1+400*p.eps*t);
	}
	*/
	/* // True solution = initial solution u0(x-a*t)
	   uTrue(vID, 0, k) = std::exp(-100*std::pow(std::fmod(x-t,1.0)-.5, 2.0));
	   uTrue(vID, 1, k) = std::exp(-100*std::pow(std::fmod(y-t,1.0)-.5, 2.0));
	   uTrue(vID, 2, k) = std::exp(-100*std::pow(std::fmod(z-t,1.0)-.5, 2.0));
	*/
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
  
  // Allocate working memory
  int nQF = mesh.nFQNodes;
  darray uCurr{dofs, nStates, mesh.nElements, 1};
  darray uInterpF{nQF, nStates, Mesh::N_FACES, mesh.nElements+mesh.nGElements, 1};
  darray uInterpV{InterpV.size(0), nStates, mesh.nElements, 1};
  darray ks{dofs, nStates, mesh.nElements, nStages};
  darray Dus{dofs, nStates, mesh.nElements, Mesh::DIM};
  darray DuInterpF{nQF, nStates, Mesh::N_FACES, mesh.nElements+mesh.nGElements, Mesh::DIM};
  darray DuInterpV{InterpV.size(0), nStates, mesh.nElements, Mesh::DIM};
  darray uTrue{dofs, nStates, mesh.nElements, 1};
  
  darray kes{timesteps};
  darray keDissipation{timesteps-2};
  
  int nBElems = max(mesh.mpiNBElems);
  darray toSend{nQF, nStates, nBElems, Mesh::DIM, MPIUtil::N_FACES};
  darray toRecv{nQF, nStates, nBElems, Mesh::DIM, MPIUtil::N_FACES};
  /**
     Requests for use in MPI sends/receives during rk4Rhs()
     2*face == send, 2*face+1 == recv
  */
  MPI_Request rk4Reqs[2*MPIUtil::N_FACES];
  
  exit(0);
  
  if (mpi.rank == mpi.ROOT) {
    std::cout << "Time stepping until tf = " << tf << std::endl;
  }
  
  auto startTime = std::chrono::high_resolution_clock::now();
  
  // Loop over time steps
  for (int iStep = 0; iStep < timesteps; ++iStep) {
    
    if (mpi.rank == mpi.ROOT) {
      std::cout << "time = " << iStep*dt << std::endl;
    }
    
    if (iStep % dtSnaps == 0) {
      if (mpi.rank == mpi.ROOT) {
	auto endTime = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = endTime-startTime;
	std::cout << "Saving snapshot " << iStep/dtSnaps << "...\n";
	std::cout << "Elapsed time so far = " << elapsed.count() << std::endl;
      }
      
      // Output snapshot files
      bool success = initXYZVFile("output/xyzu.txt", Mesh::DIM, iStep/dtSnaps, "u", nStates);
      if (!success)
	exit(-1);
      success = exportToXYZVFile("output/xyzu.txt", iStep/dtSnaps, mesh.globalCoords, u);
      if (!success)
	exit(-1);
      
      /* // TODO: debugging for convection problem
      trueSolution(uTrue, iStep*dt);
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
      
      /*// TODO: debugging - exit after 10 snapshots
      if (iStep/dtSnaps == 10) {
	if (mpi.rank == mpi.ROOT) {
	  std::cout << "exiting for debugging purposes...\n";
	}
	exit(0);
      } */
      
      
    }
    
    // Use RK4 to compute k values at each stage
    for (int istage = 0; istage < nStages; ++istage) {
      
      // Updates uCurr = u+dt*a(s,s-1)*ks(:,s)
      rk4UpdateCurr(uCurr, diagA, ks, istage);
      
      // Updates ks(:,istage) = rhs(uCurr) based on DG method
      rk4Rhs(uCurr, uInterpF, uInterpV, Dus, DuInterpF, DuInterpV,
	     toSend, toRecv, rk4Reqs, ks, istage);
      
    }
    
    // Use RK4 to move to next time step
    for (int istage = 0; istage < nStages; ++istage) {
      for (int iK = 0; iK < mesh.nElements; ++iK) {
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iN = 0; iN < dofs; ++iN) {
	    u(iN,iS,iK) += dt*b(istage)*ks(iN,iS,iK,istage);
	    
	    if (istage == nStages-1 && iK == 0 && iS == 4 && mpi.rank == mpi.ROOT) {
	      std::cout << "u[" << iN << "] = " << u(iN,iS,iK) << std::endl;
	    }
	  }
	}
      }
    }
    
    kes(iStep) = computeKE(uInterpV);
    if (iStep > 1) {
      keDissipation(iStep-2) = -(kes(iStep) - kes(iStep-2))/(2*dt);
    }
    
  }
  
  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = endTime-startTime;
  if (mpi.rank == mpi.ROOT) {
    std::cout << "Finished time stepping. Time elapsed = " << elapsed.count() << std::endl;
  
    std::cout << "Outputting kinetic energy dissipation for Navier-Stokes: " << std::endl;
    std::cout << keDissipation << std::endl;
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
   Computes the RHS of Runge Kutta method for use with Local DG method.
   Updates ks(:,istage) with rhs evaluated at uCurr
*/
void Solver::rk4Rhs(const darray& uCurr, darray& uInterpF, darray& uInterpV, 
		    darray& Dus, darray& DuInterpF, darray& DuInterpV, 
		    darray& toSend, darray& toRecv, MPI_Request * rk4Reqs, darray& ks, int istage) const {
  
  // Interpolate uCurr once
  interpolate(uCurr, uInterpF, uInterpV, toSend, toRecv, rk4Reqs, 1);
  
  // First solve for the Dus in each dimension according to:
  // Du_l = Mel\(-S_l*u + fluxesL(u))
  Dus.fill(0.0);
  localDGFlux(uInterpF, Dus);
  for (int l = 0; l < Mesh::DIM; ++l) {
    
    // Du_l = -S_l*u + Fl(u)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
		dofs, nStates*mesh.nElements, dofs, -1.0, &Sels(0,0,l), dofs,
		uCurr.data(), dofs, 1.0, &Dus(0,0,0,l), dofs);
    
    // Du_l = Mel\Du_l
    LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'U', dofs, nStates*mesh.nElements,
		   Mel.data(), dofs, Mipiv.data(), &Dus(0,0,0,l), dofs);
    
  }
  
  // Interpolate Dus once
  interpolate(Dus, DuInterpF, DuInterpV, toSend, toRecv, rk4Reqs, Mesh::DIM);
  
  // Now compute ks(:,istage) from uCurr and these Dus according to:
  // ks(:,istage) = Mel\( K*fc(u) + K*fv(u,Dus) - Fc(u) - Fv(u,Dus) )
  
  darray residual{&ks(0,0,0,istage), dofs, nStates, mesh.nElements};
  residual.fill(0.0);
  
  convectDGFlux(uInterpF, residual);
  viscousDGFlux(uInterpF, DuInterpF, residual);
  
  // ks(:,istage) = -Fc(u)-Fv(u,Dus)
  cblas_dscal(dofs*nStates*mesh.nElements, -1.0, residual.data(), 1);
  
  // ks(:,istage) += Kc*fc(u)
  convectDGVolume(uInterpV, residual);
  
  // ks(:,istage) += Kv*fv(u)
  viscousDGVolume(uInterpV, DuInterpV, residual);
  
  // ks(:,istage) = Mel\ks(:,istage)
  LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'U', dofs, nStates*mesh.nElements,
		 Mel.data(), dofs, Mipiv.data(), residual.data(), dofs);
  
}

/**
   Interpolates u/Du on faces to 2D quadrature points and stores in u/DuInterpF.
   Interpolates u/Du onto 3D quadrature points and stores in u/DuInterpV.
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
  int nQV = InterpV.size(0);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      nQV, nStates*mesh.nElements*dim, dofs, 1.0, InterpV.data(), dofs,
	      curr.data(), dofs, 0.0, toInterpV.data(), nQV);
  
  // TODO: move after all interior elements have been computed on
  mpiEndComm(toInterpF, dim, toRecv, rk4Reqs);
  
}


/**
   Local DG Flux: Computes Fl(u) for use in the local DG formulation of second-order terms.
   Uses a downwind formulation for the flux term. 
   Updates residuals arrays with added flux.
*/
void Solver::localDGFlux(const darray& uInterpF, darray& residuals) const {
  
  int nFN = mesh.nFNodes;
  int nQF = mesh.nFQNodes;
  
  // Initialize flux along faces
  darray fStar{nQF, nStates};
  darray faceContributions{nFN, nStates};
  
  // Loop over all elements
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    
    // There are l equations to handle in this flux term
    for (int l = 0; l < Mesh::DIM; ++l) {
      
      // For every face, compute fstar = fl*(InterpF*u)
      for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
	
	auto nF = mesh.eToF(iF, iK);
	auto nK = mesh.eToE(iF, iK);
	
	auto normalK = mesh.normals(l, iF, iK);
	
	// Must compute nStates of these flux integrals per face
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iFQ = 0; iFQ < nQF; ++iFQ) {
	    auto uK = uInterpF(iFQ, iS, iF, iK);
	    auto uN = uInterpF(iFQ, iS, nF, nK);
	    
	    fStar(iFQ, iS) = numericalFluxL(uK, uN, normalK);
	  }
	}
	
	// Flux contribution = KelsF(:,l)'*fstar
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
		    nFN, nStates, nQF, 1.0, &KelsF(0,0,iF/2), nQF, 
		    fStar.data(), nQF, 
		    0.0, faceContributions.data(), nFN);
	
	// Add up face contributions into global residuals array
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iFN = 0; iFN < nFN; ++iFN) {
	    residuals(mesh.efToN(iFN,iF), iS, iK, l) += faceContributions(iFN, iS);
	  }
	}
	
      }
    }
  }
  
}

/**
   Convect DG Flux: Computes Fc(u) for use in the Local DG formulation of the 
   1st-order convection term.
   Uses upwinding for the numerical flux. 
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
      
      darray normalK{&mesh.normals(0, iF, iK), 3};
      
      for (int iFQ = 0; iFQ < nQF; ++iFQ) {
	
	// Get all state variables at this point
	for (int iS = 0; iS < nStates; ++iS) {
	  uK(iS) = uInterpF(iFQ, iS, iF, iK);
	  uN(iS) = uInterpF(iFQ, iS, nF, nK);
	}
	// Compute fluxes = fc*(u+,u-)'*n
	numericalFluxC(uN, uK, normalK, fluxes);
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
  
  int nQV = InterpV.size(0);
  
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
      fluxC(uK, fluxes);
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
   Viscous DG Flux: Computes Fv(u) for use in the Local DG formulation of the 
   2nd-order diffusion term.
   Uses upwinding for the numerical flux. 
   Updates residual variable with added flux
*/
void Solver::viscousDGFlux(const darray& uInterpF, const darray& DuInterpF, darray& residual) const {
  
  int nFN = mesh.nFNodes;
  int nQF = mesh.nFQNodes;
  
  // Initialize flux along faces
  darray fStar{nQF, nStates};
  darray faceContributions{nFN, nStates};
  
  darray fluxes{nStates};
  darray uK{nStates};
  darray uN{nStates};
  darray DuK{nStates, Mesh::DIM};
  darray DuN{nStates, Mesh::DIM};
  
  // Loop over all elements
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    
    for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
      
      // For every face, compute fstar = fv*(InterpF*u,InterpF*Du)
      auto nF = mesh.eToF(iF, iK);
      auto nK = mesh.eToE(iF, iK);
      
      darray normalK{&mesh.normals(0, iF, iK), 3};
      
      for (int iFQ = 0; iFQ < nQF; ++iFQ) {
	
	// Copy all state variables at this point into uK,uN,DuK,DuN
	for (int iS = 0; iS < nStates; ++iS) {
	  uK(iS) = uInterpF(iFQ, iS, iF, iK);
	  uN(iS) = uInterpF(iFQ, iS, nF, nK);
	}
	for (int l = 0; l < Mesh::DIM; ++l) {
	  for (int iS = 0; iS < nStates; ++iS) {
	    DuK(iS,l) = DuInterpF(iFQ, iS, iF, iK, l);
	    DuN(iS,l) = DuInterpF(iFQ, iS, nF, nK, l);
	  }
	}
	// Compute fluxes = fv*(u+,u-,Du+,Du-)'*n
	numericalFluxV(uN, uK, DuN, DuK, normalK, fluxes);
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

/** Evaluates the volume integral term for viscosity in the RHS */
void Solver::viscousDGVolume(const darray& uInterpV, const darray& DuInterpV, darray& residual) const {
  
  int nQV = InterpV.size(0);
  
  // Contains all flux information
  darray fv{nQV, nStates, mesh.nElements, Mesh::DIM};
  
  // Temporary arrays for computing fluxes
  darray fluxes{nStates, Mesh::DIM};
  darray uK{nStates};
  darray DuK{nStates, Mesh::DIM};
  
  // Loop over all elements, points
  for (int k = 0; k < mesh.nElements; ++k) {
    for (int iQ = 0; iQ < nQV; ++iQ) {
      
      // Copy data into uK,DuK
      for (int iS = 0; iS < nStates; ++iS) {
	uK(iS) = uInterpV(iQ,iS,k);
      }
      for (int l = 0; l < Mesh::DIM; ++l) {
	for (int iS = 0; iS < nStates; ++iS) {
	  DuK(iS,l) = DuInterpV(iQ,iS,k,l);
	}
      }
      
      // Compute fluxes
      fluxV(uK, DuK, fluxes);
      // Copy negative data into fv
      for (int l = 0; l < Mesh::DIM; ++l) {
	for (int iS = 0; iS < nStates; ++iS) {
	  fv(iQ,iS,k,l) = -fluxes(iS, l);
	}
      }
      
    }
  }
  
  // TODO: is it faster to do matrix vector multiplication instead of having to reorder data into huge fc array?
  
  // residual += Kels(:, l)*fc_l(uInterpV)
  for (int l = 0; l < Mesh::DIM; ++l) {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		dofs, nStates*mesh.nElements, nQV, 1.0, &Kels(0,0,l), nQV,
		&fv(0,0,0,l), nQV, 1.0, residual.data(), dofs);
  }
  
}

/** Evaluates the local DG flux function for this PDE using a positive-looking switch */
inline double Solver::numericalFluxL(double uK, double uN, double normalK) const {
  
  auto fK = uK;
  auto fN = uN;
  
  bool kWins;
  /* // True switch requires the entire normal vector
  double sum = 0.0;
  for (int l = 0; l < Mesh::DIM; ++l) {
    sum += normalK(l);
  }
  kWins = (sum > 0);
  */
  kWins = (normalK > 0);
  
  return (kWins ? fK : fN)*normalK;
  
}


/**
   Evaluates the viscous numerical flux function for this PDE for all states, 
   dotted with the normal, storing output in fluxes.
*/
void Solver::numericalFluxV(const darray& uN, const darray& uK, 
			    const darray& DuN, const darray& DuK, 
			    const darray& normalK, darray& fluxes) const {
  
  darray Flux{nStates, Mesh::DIM};
  
  // Use current element's uK for flux
  // Use negative of switch from numericalFluxL for Du
  bool kWins;
  double sum = 0.0;
  for (int l = 0; l < Mesh::DIM; ++l) {
    sum += normalK(l);
    
  }
  kWins = (sum > 0);
  
  if (!kWins) {
    fluxV(uK, DuK, Flux);
  }
  else {
    fluxV(uK, DuN, Flux);
  }
  
  fluxes.fill(0.0);
  for (int l = 0; l < Mesh::DIM; ++l) {
    for (int iS = 0; iS < nStates; ++iS) {
      fluxes(iS) -= Flux(iS, l)*normalK(l);
    }
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

/** Computes the kinetic energy for a given solution to Navier-Stokes */
double Solver::computeKE(const darray& uInterpV) const {
  double globalkEnergy = 0.0;
  double localkEnergy = 0.0;
  
  darray xQ, wQ;
  int nQV = gaussQuad2D(2*order, xQ, wQ);
  
  darray alpha{Mesh::DIM};
  alpha(0) = mesh.minDX/2.0;
  alpha(1) = mesh.minDY/2.0;
  double Jacobian = alpha(0)*alpha(1);
  
  darray kes{mesh.nElements};
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int iFQ = 0; iFQ < nQV; ++iFQ) {
      double rho2v2 = 0.0;
      for (int l = 0; l < Mesh::DIM; ++l) {
	rho2v2 += u(iFQ,l+1,iK)*u(iFQ,l+1,iK);
      }
      kes(iK) += Jacobian*wQ(iFQ)*rho2v2/(2*u(iFQ,0,iK));
    }
    
    localkEnergy += kes(iK);
  }
  
  MPI_Allreduce(&localkEnergy, &globalkEnergy, 1, MPI_DOUBLE, MPI_SUM, mpi.cartComm);
  
  globalkEnergy = globalkEnergy/(p.rho0*std::pow(2*M_PI*p.L, 3.0));
  
  return globalkEnergy;
}




/** Evaluates the actual convection flux function for the PDE */
void Solver::fluxC(const darray& uK, darray& fluxes) const {
  
  /* // Convection-diffusion along each dimension
  fluxes.fill(0.0);
  for (int l = 0; l < Mesh::DIM; ++l) {
    fluxes(l,l) = uK(l);
  }
  */
  
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
  
}

/** Evaluates the actual viscosity flux function for the PDE */
void Solver::fluxV(const darray& uK, const darray& DuK, darray& fluxes) const {
  
  /* // Convection-diffusion along each dimension
  fluxes.fill(0.0);
  for (int l = 0; l < Mesh::DIM; ++l) {
    fluxes(l,l) = p.eps*DuK(l,l);
  }
  */
  
  // Fv2d for Navier-Stokes
  double t2;
  double t30;
  double t16;
  double t3;
  double t17;
  double t5;
  double t6;
  double t19;
  double t8;
  double t12;
  double t45;
  double t25;
  double t40;
  double t32;
  double t46;
  double t38;
  double t21;
  double t9;
  double t13;
  double t47;
  double t49;
  double t57;
  fluxes[0] = 0.0e0;
  t2 = DuK[0];
  t3 = uK[1];
  t5 = uK[0];
  t6 = 0.1e1 / t5;
  t8 = DuK[1] - t2 * t3 * t6;
  t9 = t8 * t6;
  t12 = DuK[4];
  t13 = uK[2];
  t16 = DuK[6] - t12 * t13 * t6;
  t17 = t16 * t6;
  t19 = 0.4e1 / 0.3e1 * t9 - 0.2e1 / 0.3e1 * t17;
  t21 = 0.1e1 / p.pars[0];
  fluxes[1] = t19 * t21;
  t25 = DuK[2] - t2 * t13 * t6;
  t30 = DuK[5] - t12 * t3 * t6;
  t32 = t25 * t6 + t30 * t6;
  fluxes[2] = t32 * t21;
  t38 = 0.1e1 / p.pars[1];
  t40 = uK[3];
  t45 = t5 * t5;
  t46 = 0.1e1 / t45;
  t47 = t3 * t46;
  t49 = t13 * t46;
  fluxes[3] = (t19 * t3 * t6 + t32 * t13 * t6 + 0.7e1 / 0.5e1 * t38 * ((DuK[3] - t2 * t40 * t6) * t6 - t47 * t8 - t49 * t25)) * t21;
  fluxes[4] = 0.0e0;
  fluxes[5] = fluxes[2];
  t57 = 0.4e1 / 0.3e1 * t17 - 0.2e1 / 0.3e1 * t9;
  fluxes[6] = t57 * t21;
  fluxes[7] = (t32 * t3 * t6 + t57 * t13 * t6 + 0.7e1 / 0.5e1 * t38 * ((DuK[7] - t12 * t40 * t6) * t6 - t47 * t30 - t49 * t16)) * t21;

}

/**
   Evaluates the convection numerical flux function for this PDE for all states, 
   dotted with the normal, storing output in fluxes.
*/
void Solver::numericalFluxC(const darray& uN, const darray& uK, 
			    const darray& normalK, darray& fluxes) const {
  
  /* // Upwinding for convection-diffusion in 3 dimensions
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
  
}
