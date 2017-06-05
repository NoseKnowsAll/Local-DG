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
  Interp2D{},
  Interp3D{},
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
  dt = 0.01*std::min(std::min(mesh.minDX, mesh.minDY), mesh.minDZ)/(maxVel*(2*order+1));
  // Change tf = 10*tc in order to visualize most turbulent structures
  //tf = 10*p.tc;
  // Ensure we will exactly end at tf
  timesteps = std::ceil(tf/dt);
  dt = tf/timesteps;
  
  // Initialize nodes within elements
  chebyshev3D(order, refNodes);
  dofs = refNodes.size(1)*refNodes.size(2)*refNodes.size(3);
  mesh.setupNodes(refNodes, order);
  mpi.initDatatype(mesh.nFQNodes);
  mpi.initFaces(Mesh::N_FACES);
  
  nStates = 5;
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
  int nQ2D   = gaussQuad2D(2*order, xQ2D, wQ2D);
  
  interpolationMatrix2D(cheby2D, xQ2D, Interp2D);
  
  // 3D interpolation matrix for use on elements
  darray cheby3D;
  chebyshev3D(order, cheby3D);
  darray xQ3D, wQ3D;
  int nQ3D   = gaussQuad3D(2*order, xQ3D, wQ3D);
  
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
  lapack_int ipiv[dofs];
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
  for (int ipz = 0; ipz <= order; ++ipz) {
    for (int ipy = 0; ipy <= order; ++ipy) {
      for (int ipx = 0; ipx <= order; ++ipx) {
	for (int iQ = 0; iQ < sizeQ; ++iQ) {
	  polyQuad(iQ, ipx,ipy,ipz) = phiQ(iQ, ipx,ipy,ipz)*wQ(iQ);
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
	      dofs, dofs, sizeQ, Jacobian, polyQuad.data(), sizeQ, 
	      phiQ.data(), sizeQ, 0.0, Mel.data(), dofs);
  Mipiv.realloc(dofs);
  // Mel overwritten with U*D*U'
  info = LAPACKE_dsytrf(LAPACK_COL_MAJOR, 'U', dofs,
			Mel.data(), dofs, Mipiv.data());
  
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
  darray xQ2D, wQ2D;
  int fSizeQ = gaussQuad2D(2*order, xQ2D, wQ2D);
  int fDofs = (order+1)*(order+1);
  Kels2D.realloc(fSizeQ,fDofs, Mesh::DIM);
  for (int l = 0; l < Mesh::DIM; ++l) {
    double scaleL = Jacobian/alpha(l);
    for (int iDofs = 0; iDofs < fDofs; ++iDofs) {
      for (int iQ = 0; iQ < fSizeQ; ++iQ) {
	Kels2D(iQ, iDofs, l) = Interp2D(iQ,iDofs)*wQ2D(iQ)*scaleL;
      }
    }
  }
  
}

void Solver::initialCondition() {
  
  // Sin function allowing for periodic initial condition
  for (int k = 0; k < mesh.nElements; ++k) {
    for (int iz = 0; iz < order+1; ++iz) {
      for (int iy = 0; iy < order+1; ++iy) {
	for (int ix = 0; ix < order+1; ++ix) {
	  int vID = ix + iy*(order+1) + iz*(order+1)*(order+1);
	  
	  double x = mesh.globalCoords(0,vID,k);
	  double y = mesh.globalCoords(1,vID,k);
	  double z = mesh.globalCoords(2,vID,k);
	    
	  //u(vID, iS, k) = std::sin(2*x*M_PI)*std::sin(2*y*M_PI)*std::sin(2*z*M_PI);
	  /*
	  u(vID, 0, k) = std::exp(-100*std::pow(x-.5, 2.0));
	  u(vID, 1, k) = std::exp(-100*std::pow(y-.5, 2.0));
	  u(vID, 2, k) = std::exp(-100*std::pow(z-.5, 2.0));
	  */
	  
	  // p
	  double pressure = p.p0 + (p.rho0*p.V0*p.V0/16.0)
	    *(std::cos(2*x/p.L)+std::cos(2*y/p.L))*(std::cos(2*z/p.L)+2);
	  // rho
	  u(vID, 0, k) = pressure/(p.R*p.T0);
	  // rho*v_0
	  u(vID, 1, k) = p.rho0*p.V0*std::sin(x/p.L)
	    *std::cos(y/p.L)*std::cos(z/p.L);
	  // rho*v_1
	  u(vID, 2, k) = -p.rho0*p.V0*std::cos(x/p.L)
	    *std::sin(y/p.L)*std::cos(z/p.L);
	  // rho*v_2
	  u(vID, 3, k) = 0*p.rho0;
	  // rho*E
	  u(vID, 4, k) = pressure/(p.gamma-1) 
	    + (u(vID,1,k)*u(vID,1,k) + u(vID,2,k)*u(vID,2,k) + u(vID,3,k)*u(vID,3,k))/(2.0*u(vID,0,k));
	  
	}
      }
    }
  }
  
}

/** Computes the true convection solution at time t for the convection problem */
void Solver::trueSolution(darray& uTrue, double t) const {
  
  int N = 2;
  
  for (int k = 0; k < mesh.nElements; ++k) {
    for (int iz = 0; iz < order+1; ++iz) {
      for (int iy = 0; iy < order+1; ++iy) {
	for (int ix = 0; ix < order+1; ++ix) {
	  int vID = ix + iy*(order+1) + iz*(order+1)*(order+1);
	  
	  double x = mesh.globalCoords(0,vID,k);
	  double y = mesh.globalCoords(1,vID,k);
	  double z = mesh.globalCoords(2,vID,k);
	  // True solution = initial solution u0(x-a*t)
	  /*
	    uTrue(vID, iS, k) = std::sin(2*fmod(x-p.a[0]*t+5.0,1.0)*M_PI)
	    *std::sin(2*fmod(y-p.a[1]*t+5.0,1.0)*M_PI)
	    *std::sin(2*fmod(z-p.a[2]*t+5.0,1.0)*M_PI);
	  */
	  
	  /* // True solution = conv-diff
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
  int nQ2D = mesh.nFQNodes;
  darray uCurr{dofs, nStates, mesh.nElements, 1};
  darray uInterp2D{nQ2D, nStates, Mesh::N_FACES, mesh.nElements+mesh.nGElements, 1};
  darray uInterp3D{Interp3D.size(0), nStates, mesh.nElements, 1};
  darray ks{dofs, nStates, mesh.nElements, nStages};
  darray Dus{dofs, nStates, mesh.nElements, Mesh::DIM};
  darray DuInterp2D{nQ2D, nStates, Mesh::N_FACES, mesh.nElements+mesh.nGElements, Mesh::DIM};
  darray DuInterp3D{Interp3D.size(0), nStates, mesh.nElements, Mesh::DIM};
  darray uTrue{dofs, nStates, mesh.nElements, 1};
  
  darray kes{timesteps};
  darray keDissipation{timesteps-2};
  
  int nBElems = max(mesh.mpiNBElems);
  darray toSend{nQ2D, nStates, nBElems, Mesh::DIM, MPIUtil::N_FACES};
  darray toRecv{nQ2D, nStates, nBElems, Mesh::DIM, MPIUtil::N_FACES};
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
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime-startTime;
    if (elapsed.count() > 19.5*60*60) {
      if (mpi.rank == mpi.ROOT)
	std::cout << "quitting voluntarily due to time limit" << std::endl;
      
      bool success = initXYZVFile("output/xyznavier.tmp", 0, "u", nStates);
      if (!success)
	exit(-1);
      success = exportToXYZVFile("output/xyznavier.tmp", 0, mesh.globalCoords, u);
      break;
    }
    
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
      
      // TODO: outputting snapshot files
      bool success = initXYZVFile("output/xyzu.txt", iStep/dtSnaps, "u", nStates);
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
      
      /* // TODO: debugging - exit after 10 snapshots
      if (iStep/dtSnaps == 10) {
	if (mpi.rank == mpi.ROOT) {
	  std::cout << "exiting for debugging purposes...\n";
	}
	exit(0);
      }
      */
      
    }
    
    // Use RK4 to compute k values at each stage
    for (int istage = 0; istage < nStages; ++istage) {
      
      // Updates uCurr = u+dt*a(s,s-1)*ks(:,s)
      rk4UpdateCurr(uCurr, diagA, ks, istage);
      
      // Updates ks(:,istage) = rhs(uCurr) based on DG method
      rk4Rhs(uCurr, uInterp2D, uInterp3D, Dus, DuInterp2D, DuInterp3D,
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
    
    
    kes(iStep) = computeKE(uInterp3D);
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
void Solver::rk4Rhs(const darray& uCurr, darray& uInterp2D, darray& uInterp3D, 
		    darray& Dus, darray& DuInterp2D, darray& DuInterp3D, 
		    darray& toSend, darray& toRecv, MPI_Request * rk4Reqs, darray& ks, int istage) const {
  
  // Interpolate uCurr once
  interpolate(uCurr, uInterp2D, uInterp3D, toSend, toRecv, rk4Reqs, 1);
  
  // First solve for the Dus in each dimension according to:
  // Du_l = Mel\(-S_l*u + fluxesL(u))
  Dus.fill(0.0);
  localDGFlux(uInterp2D, Dus);
  for (int l = 0; l < Mesh::DIM; ++l) {
    
    // Du_l = -S_l*u + Fl(u)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
		dofs, nStates*mesh.nElements, dofs, -1.0, &Sels(0,0,l), dofs,
		uCurr.data(), dofs, 1.0, &Dus(0,0,0,l), dofs);
    
    // Du_l = Mel\Du_l
    int info = LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'U', dofs, nStates*mesh.nElements,
                             Mel.data(), dofs, Mipiv.data(), &Dus(0,0,0,l), dofs);
    
  }
  
  // Interpolate Dus once
  interpolate(Dus, DuInterp2D, DuInterp3D, toSend, toRecv, rk4Reqs, Mesh::DIM);
  
  // Now compute ks(:,istage) from uCurr and these Dus according to:
  // ks(:,istage) = Mel\( K*fc(u) + K*fv(u,Dus) - Fc(u) - Fv(u,Dus) )
  
  darray residual{&ks(0,0,0,istage), dofs, nStates, mesh.nElements};
  residual.fill(0.0);
  
  convectDGFlux(uInterp2D, residual);
  viscousDGFlux(uInterp2D, DuInterp2D, residual);
  
  // ks(:,istage) = -Fc(u)-Fv(u,Dus)
  cblas_dscal(dofs*nStates*mesh.nElements, -1.0, residual.data(), 1);
  
  // ks(:,istage) += Kc*fc(u)
  convectDGVolume(uInterp3D, residual);
  
  // ks(:,istage) += Kv*fv(u)
  viscousDGVolume(uInterp3D, DuInterp3D, residual);
  
  // ks(:,istage) = Mel\ks(:,istage)
  int info = LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'U', dofs, nStates*mesh.nElements,
			    Mel.data(), dofs, Mipiv.data(), residual.data(), dofs);
  
}

/**
   Interpolates u/Du on faces to 2D quadrature points and stores in u/DuInterp2D.
   Interpolates u/Du onto 3D quadrature points and stores in u/DuInterp3D.
*/
void Solver::interpolate(const darray& curr, darray& toInterp2D, darray& toInterp3D,
			 darray& toSend, darray& toRecv, MPI_Request * rk4Reqs, int dim) const {
  
  // First grab u on faces and pack into array uOnFaces
  int nFN = mesh.nFNodes;
  int nQ2D = mesh.nFQNodes;
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
    
    // 2D interpolation toInterp2D = Interp2D*uOnFaces
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		nQ2D, nStates*Mesh::N_FACES*mesh.nElements, nFN, 1.0, 
		Interp2D.data(), nQ2D, &onFaces(0,0,0,0,l), nFN, 
		0.0, &toInterp2D(0,0,0,0,l), nQ2D);
  }
  
  mpiStartComm(toInterp2D, dim, toSend, toRecv, rk4Reqs);
  
  // 3D interpolation toInterp3D = Interp3D*curr
  int nQ3D = Interp3D.size(0);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      nQ3D, nStates*mesh.nElements*dim, dofs, 1.0, Interp3D.data(), dofs,
	      curr.data(), dofs, 0.0, toInterp3D.data(), nQ3D);
  
  // TODO: move after all interior elements have been computed on
  mpiEndComm(toInterp2D, dim, toRecv, rk4Reqs);
  
}


/**
   Local DG Flux: Computes Fl(u) for use in the local DG formulation of second-order terms.
   Uses a downwind formulation for the flux term. 
   Updates residuals arrays with added flux.
*/
void Solver::localDGFlux(const darray& uInterp2D, darray& residuals) const {
  
  int nFN = mesh.nFNodes;
  int nQ2D = mesh.nFQNodes;
  
  // Initialize flux along faces
  darray fStar{nQ2D, nStates};
  darray faceContributions{nFN, nStates};
  
  // Loop over all elements
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    
    // There are l equations to handle in this flux term
    for (int l = 0; l < Mesh::DIM; ++l) {
      
      // For every face, compute fstar = fl*(Interp2D*u)
      for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
	
	auto nF = mesh.eToF(iF, iK);
	auto nK = mesh.eToE(iF, iK);
	
	auto normalK = mesh.normals(l, iF, iK);
	
	// Must compute nStates of these flux integrals per face
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iFQ = 0; iFQ < nQ2D; ++iFQ) {
	    auto uK = uInterp2D(iFQ, iS, iF, iK);
	    auto uN = uInterp2D(iFQ, iS, nF, nK);
	    
	    fStar(iFQ, iS) = numericalFluxL(uK, uN, normalK);
	  }
	}
	
	// Flux contribution = Kels2D(:,l)'*fstar
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
		    nFN, nStates, nQ2D, 1.0, &Kels2D(0,0,iF/2), nQ2D, 
		    fStar.data(), nQ2D, 
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
void Solver::convectDGFlux(const darray& uInterp2D, darray& residual) const {
  
  int nFN = mesh.nFNodes;
  int nQ2D = mesh.nFQNodes;
  
  // Initialize flux along faces
  darray fStar{nQ2D, nStates};
  darray faceContributions{nFN, nStates};
  
  darray fluxes{nStates};
  darray uK{nStates};
  darray uN{nStates};
  
  // Loop over all elements
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
      
      // For every face, compute fstar = fc*(Interp2D*u)
      auto nF = mesh.eToF(iF, iK);
      auto nK = mesh.eToE(iF, iK);
      
      darray normalK{&mesh.normals(0, iF, iK), 3};
      
      for (int iFQ = 0; iFQ < nQ2D; ++iFQ) {
	
	// Get all state variables at this point
	for (int iS = 0; iS < nStates; ++iS) {
	  uK(iS) = uInterp2D(iFQ, iS, iF, iK);
	  uN(iS) = uInterp2D(iFQ, iS, nF, nK);
	}
	// Compute fluxes = fc*(u+,u-)'*n
	numericalFluxC(uN, uK, normalK, fluxes);
	// Copy flux data into fStar
	for (int iS = 0; iS < nStates; ++iS) {
	  fStar(iFQ, iS) = fluxes(iS);
	}
	
      }
      
      // Flux contribution = Kels2D(:,l)'*fstar
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
		  nFN, nStates, nQ2D, 1.0, &Kels2D(0,0,iF/2), nQ2D, 
		  fStar.data(), nQ2D, 
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
void Solver::convectDGVolume(const darray& uInterp3D, darray& residual) const {
  
  int nQ3D = Interp3D.size(0);
  
  // Contains all flux information
  darray fc{nQ3D, nStates, mesh.nElements, Mesh::DIM};
  
  // Temporary arrays for computing fluxes
  darray fluxes{nStates, Mesh::DIM};
  darray uK{nStates};
  
  // Loop over all elements, points
  for (int k = 0; k < mesh.nElements; ++k) {
    for (int iQ = 0; iQ < nQ3D; ++iQ) {
      
      // Copy data into uK
      for (int iS = 0; iS < nStates; ++iS) {
	uK(iS) = uInterp3D(iQ,iS,k);
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
  
  // residual += Kels(:, l)*fc_l(uInterp3D)
  for (int l = 0; l < Mesh::DIM; ++l) {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		dofs, nStates*mesh.nElements, nQ3D, 1.0, &Kels(0,0,l), nQ3D,
		&fc(0,0,0,l), nQ3D, 1.0, residual.data(), dofs);
  }
  
}


/**
   Viscous DG Flux: Computes Fv(u) for use in the Local DG formulation of the 
   2nd-order diffusion term.
   Uses upwinding for the numerical flux. 
   Updates residual variable with added flux
*/
void Solver::viscousDGFlux(const darray& uInterp2D, const darray& DuInterp2D, darray& residual) const {
  
  int nFN = mesh.nFNodes;
  int nQ2D = mesh.nFQNodes;
  
  // Initialize flux along faces
  darray fStar{nQ2D, nStates};
  darray faceContributions{nFN, nStates};
  
  darray fluxes{nStates};
  darray uK{nStates};
  darray uN{nStates};
  darray DuK{nStates, Mesh::DIM};
  darray DuN{nStates, Mesh::DIM};
  
  // Loop over all elements
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    
    for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
      
      // For every face, compute fstar = fv*(Interp2D*u,Interp2D*Du)
      auto nF = mesh.eToF(iF, iK);
      auto nK = mesh.eToE(iF, iK);
      
      darray normalK{&mesh.normals(0, iF, iK), 3};
      
      for (int iFQ = 0; iFQ < nQ2D; ++iFQ) {
	
	// Copy all state variables at this point into uK,uN,DuK,DuN
	for (int iS = 0; iS < nStates; ++iS) {
	  uK(iS) = uInterp2D(iFQ, iS, iF, iK);
	  uN(iS) = uInterp2D(iFQ, iS, nF, nK);
	}
	for (int l = 0; l < Mesh::DIM; ++l) {
	  for (int iS = 0; iS < nStates; ++iS) {
	    DuK(iS,l) = DuInterp2D(iFQ, iS, iF, iK, l);
	    DuN(iS,l) = DuInterp2D(iFQ, iS, nF, nK, l);
	  }
	}
	// Compute fluxes = fv*(u+,u-,Du+,Du-)'*n
	numericalFluxV(uN, uK, DuN, DuK, normalK, fluxes);
	// Copy flux data into fStar
	for (int iS = 0; iS < nStates; ++iS) {
	  fStar(iFQ, iS) = fluxes(iS);
	}
	
      }
      
      // Flux contribution = Kels2D(:,l)'*fstar
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
		  nFN, nStates, nQ2D, 1.0, &Kels2D(0,0,iF/2), nQ2D, 
		  fStar.data(), nQ2D, 
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
void Solver::viscousDGVolume(const darray& uInterp3D, const darray& DuInterp3D, darray& residual) const {
  
  int nQ3D = Interp3D.size(0);
  
  // Contains all flux information
  darray fv{nQ3D, nStates, mesh.nElements, Mesh::DIM};
  
  // Temporary arrays for computing fluxes
  darray fluxes{nStates, Mesh::DIM};
  darray uK{nStates};
  darray DuK{nStates, Mesh::DIM};
  
  // Loop over all elements, points
  for (int k = 0; k < mesh.nElements; ++k) {
    for (int iQ = 0; iQ < nQ3D; ++iQ) {
      
      // Copy data into uK,DuK
      for (int iS = 0; iS < nStates; ++iS) {
	uK(iS) = uInterp3D(iQ,iS,k);
      }
      for (int l = 0; l < Mesh::DIM; ++l) {
	for (int iS = 0; iS < nStates; ++iS) {
	  DuK(iS,l) = DuInterp3D(iQ,iS,k,l);
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
  
  // residual += Kels(:, l)*fc_l(uInterp3D)
  for (int l = 0; l < Mesh::DIM; ++l) {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		dofs, nStates*mesh.nElements, nQ3D, 1.0, &Kels(0,0,l), nQ3D,
		&fv(0,0,0,l), nQ3D, 1.0, residual.data(), dofs);
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
   Assumes interpolated is of size (nQ2D, nStates, Mesh::N_FACES, nElements+nGElements, dim)
*/
void Solver::mpiStartComm(const darray& interpolated, int dim, darray& toSend, darray& toRecv, MPI_Request * rk4Reqs) const {
  
  int nQ2D = mesh.nFQNodes;
  
  // Pack face data to send
  for (int iF = 0; iF < MPIUtil::N_FACES; ++iF) {
    
    for (int l = 0; l < dim; ++l) {
      for (int bK = 0; bK < mesh.mpiNBElems(iF); ++bK) {
	int iK = mesh.mpibeToE(bK, iF);
	
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iQ = 0; iQ < nQ2D; ++iQ) {
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
   MPI communication: Finalizes nonblocking Sends/Recvs of uInterp2D.
   Then unpacks data from recv arrays into interpolated.
   Assumes interpolated is of size (nQ2D, nStates, Mesh::N_FACES, nElements, dim)
*/
void Solver::mpiEndComm(darray& interpolated, int dim, const darray& toRecv, MPI_Request * rk4Reqs) const {
  
  int nQ2D = mesh.nFQNodes;
  
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
	  for (int iQ = 0; iQ < nQ2D; ++iQ) {
	    interpolated(iQ, iS, nF, nK, l) = toRecv(iQ, iS, bK, l, iF);
	  }
	}
      }
    }
    
  }
  
}





//////////////////////////////////////////////////////////////////////////////////

/** Computes the kinetic energy for a given solution to Navier-Stokes */
double Solver::computeKE(const darray& uInterp3D) const {
  double globalkEnergy = 0.0;
  double localkEnergy = 0.0;
  
  darray xQ, wQ;
  int nQ3D = gaussQuad3D(2*order, xQ, wQ);
  
  darray alpha{Mesh::DIM};
  alpha(0) = mesh.minDX/2.0;
  alpha(1) = mesh.minDY/2.0;
  alpha(2) = mesh.minDZ/2.0;
  double Jacobian = alpha(0)*alpha(1)*alpha(2);
  
  darray kes{mesh.nElements};
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int iFQ = 0; iFQ < nQ3D; ++iFQ) {
      kes(iK) += Jacobian*wQ(iFQ)*(u(iFQ,1,iK)*u(iFQ,1,iK) + u(iFQ,2,iK)*u(iFQ,2,iK) + u(iFQ,3,iK)*u(iFQ,3,iK))/(2*u(iFQ,0,iK));
    }
    
    localkEnergy += kes(iK);
  }
  
  MPI_Allreduce(&localkEnergy, &globalkEnergy, 1, MPI_DOUBLE, MPI_SUM, mpi.cartComm);
  
  globalkEnergy = globalkEnergy/(p.rho0*std::pow(2*M_PI*p.L, 3.0));
  
  return globalkEnergy;
}




/** Evaluates the actual convection flux function for the PDE */
void Solver::fluxC(const darray& uK, darray& fluxes) const {
  
  /* // Convection-diffusion along 3 dimensions
  fluxes.fill(0.0);
  fluxes(0,0) = uK(0);
  fluxes(1,1) = uK(1);
  fluxes(2,2) = uK(2);
  */
  
  // Fi3d for Navier-Stokes
  double t17;
  double t10;
  double t9;
  double t13;
  double t3;
  double t5;
  double t6;
  double t1;
  double t7;
  double t8;
  fluxes[0] = uK[1];
  t1 = std::pow(fluxes[0], 2.0);
  t3 = 1.0 / uK[0];
  t5 = uK[4];
  t6 = 0.2e1 / 0.5e1 * t5;
  t7 = uK[2];
  t8 = t7 * t7;
  t9 = uK[3];
  t10 = t9 * t9;
  t13 = (t1 + t8 + t10) * t3 / 0.5e1;
  fluxes[1] = t1 * t3 + t6 - t13;
  fluxes[2] = fluxes[0] * t7 * t3;
  fluxes[3] = fluxes[0] * t9 * t3;
  t17 = p.gamma * t5 - t13;
  fluxes[4] = fluxes[0] * t17 * t3;
  fluxes[5] = t7;
  fluxes[6] = fluxes[2];
  fluxes[7] = t8 * t3 + t6 - t13;
  fluxes[8] = fluxes[5] * t9 * t3;
  fluxes[9] = fluxes[5] * t17 * t3;
  fluxes[10] = t9;
  fluxes[11] = fluxes[3];
  fluxes[12] = fluxes[8];
  fluxes[13] = t10 * t3 + t6 - t13;
  fluxes[14] = fluxes[10] * t17 * t3;
  
}

/** Evaluates the actual viscosity flux function for the PDE */
void Solver::fluxV(const darray& uK, const darray& DuK, darray& fluxes) const {
  
  /* // Convection-diffusion along 3 dimension
  fluxes.fill(0.0);
  fluxes(0,0) = p.eps*DuK(0,0);
  fluxes(1,1) = p.eps*DuK(1,1);
  fluxes(2,2) = p.eps*DuK(2,2);
  */
  
  // Fv3d for Navier-Stokes
  double t91;
  double t70;
  double t20;
  double t12;
  double t5;
  double t6;
  double t38;
  double t40;
  double t49;
  double t66;
  double t67;
  double t80;
  double t33;
  double t61;
  double t79;
  double t13;
  double t111;
  double t21;
  double t17;
  double t8;
  double t9;
  double t72;
  double t3;
  double t18;
  double t59;
  double t24;
  double t68;
  double t2;
  double t51;
  double t44;
  double t89;
  double t25;
  double t29;
  double t26;
  double t27;
  double t84;
  double t16;
  fluxes[0] = 0.0e0;
  t2 = DuK[0];
  t3 = uK[1];
  t5 = uK[0];
  t6 = 0.1e1 / t5;
  t8 = DuK[1] - t2 * t3 * t6;
  t9 = t8 * t6;
  t12 = DuK[5];
  t13 = uK[2];
  t16 = DuK[7] - t12 * t13 * t6;
  t17 = t16 * t6;
  t18 = 0.2e1 / 0.3e1 * t17;
  t20 = DuK[10];
  t21 = uK[3];
  t24 = DuK[13] - t20 * t21 * t6;
  t25 = t24 * t6;
  t26 = 0.2e1 / 0.3e1 * t25;
  t27 = 0.4e1 / 0.3e1 * t9 - t18 - t26;
  t29 = 0.1e1 / p.pars[0];
  fluxes[1] = t27 * t29;
  t33 = DuK[2] - t2 * t13 * t6;
  t38 = DuK[6] - t12 * t3 * t6;
  t40 = t33 * t6 + t38 * t6;
  fluxes[2] = t40 * t29;
  t44 = DuK[3] - t2 * t21 * t6;
  t49 = DuK[11] - t20 * t3 * t6;
  t51 = t44 * t6 + t49 * t6;
  fluxes[3] = t51 * t29;
  t59 = 0.1e1 / p.pars[1];
  t61 = uK[4];
  t66 = t5 * t5;
  t67 = 0.1e1 / t66;
  t68 = t3 * t67;
  t70 = t13 * t67;
  t72 = t21 * t67;
  fluxes[4] = (t27 * t3 * t6 + t40 * t13 * t6 + t51 * t21 * t6 + p.gamma * t59 * ((DuK[4] - t2 * t61 * t6) * t6 - t68 * t8 - t70 * t33 - t72 * t44)) * t29;
  fluxes[5] = 0.0e0;
  fluxes[6] = fluxes[2];
  t79 = 0.2e1 / 0.3e1 * t9;
  t80 = 0.4e1 / 0.3e1 * t17 - t79 - t26;
  fluxes[7] = t80 * t29;
  t84 = DuK[8] - t12 * t21 * t6;
  t89 = DuK[12] - t20 * t13 * t6;
  t91 = t84 * t6 + t89 * t6;
  fluxes[8] = t91 * t29;
  fluxes[9] = (t40 * t3 * t6 + t80 * t13 * t6 + t91 * t21 * t6 + p.gamma * t59 * ((DuK[9] - t12 * t61 * t6) * t6 - t68 * t38 - t70 * t16 - t72 * t84)) * t29;
  fluxes[10] = 0.0e0;
  fluxes[11] = fluxes[3];
  fluxes[12] = fluxes[8];
  t111 = 0.4e1 / 0.3e1 * t25 - t79 - t18;
  fluxes[13] = t111 * t29;
  fluxes[14] = (t51 * t3 * t6 + t91 * t13 * t6 + t111 * t21 * t6 + p.gamma * t59 * ((DuK[14] - t20 * t61 * t6) * t6 - t68 * t49 - t70 * t89 - t72 * t24)) * t29;
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
  
  // roe3d from Navier-Stokes
  double t105;
  double t138;
  double t8;
  double t10;
  double t44;
  double t27;
  double t28;
  double t35;
  double t186;
  double t76;
  double t55;
  double t56;
  double t58;
  double t23;
  double t24;
  double t18;
  double t125;
  double t126;
  double t128;
  double t129;
  double t131;
  double t132;
  double t134;
  double t135;
  double t137;
  double t77;
  double t78;
  double t81;
  double t48;
  double t108;
  double t113;
  double t117;
  double t120;
  double t121;
  double t122;
  double t123;
  double t149;
  double t46;
  double t47;
  double t50;
  double t63;
  double t64;
  double t2;
  double t3;
  double t152;
  double t155;
  double t158;
  double t160;
  double t96;
  double t66;
  double t67;
  double t68;
  double t165;
  double t167;
  double t168;
  double t1;
  double t69;
  double t174;
  double t13;
  double t5;
  double t89;
  double t139;
  double t179;
  double t31;
  double t32;
  double t21;
  double t22;
  double t84;
  double t15;
  double t39;
  double t75;
  double t99;
  double t72;
  double t146;
  double t100;
  double t101;
  double t102;
  t1 = uN[0];
  t2 = uN[1];
  t3 = 0.1e1 / t1;
  t5 = normalK[0];
  t8 = uN[2];
  t10 = normalK[1];
  t13 = uN[3];
  t15 = normalK[2];
  t18 = 0.10e1 * t2 * t3 * t5 + 0.10e1 * t8 * t3 * t10 + 0.10e1 * t13 * t3 * t15;
  t21 = uK[0];
  t22 = uK[1];
  t23 = 0.1e1 / t21;
  t24 = t22 * t23;
  t27 = uK[2];
  t28 = t27 * t23;
  t31 = uK[3];
  t32 = t31 * t23;
  t35 = 0.10e1 * t24 * t5 + 0.10e1 * t10 * t28 + 0.10e1 * t32 * t15;
  t39 = std::sqrt(t1 * t23);
  t44 = 0.100e1 * t39 * t2 * t3 + 0.10e1 * t24;
  t46 = 0.10e1 * t39 + 0.10e1;
  t47 = 0.1e1 / t46;
  t48 = t44 * t47;
  t50 = 0.10e1 * t48 * t5;
  t55 = 0.100e1 * t39 * t8 * t3 + 0.10e1 * t28;
  t56 = t55 * t47;
  t58 = 0.10e1 * t56 * t10;
  t63 = 0.100e1 * t39 * t13 * t3 + 0.10e1 * t32;
  t64 = t63 * t47;
  t66 = 0.10e1 * t64 * t15;
  t67 = t50 + t58 + t66;
  t68 = std::abs(t67);
  t69 = t1 - t21;
  t72 = uN[4];
  t75 = 0.4e0 * t72;
  t76 = t2 * t2;
  t77 = t1 * t1;
  t78 = 0.1e1 / t77;
  t81 = t8 * t8;
  t84 = t13 * t13;
  t89 = 0.20e0 * t1 * (0.100e1 * t76 * t78 + 0.100e1 * t81 * t78 + 0.100e1 * t84 * t78);
  t96 = uK[4];
  t99 = 0.4e0 * t96;
  t100 = t22 * t22;
  t101 = t21 * t21;
  t102 = 0.1e1 / t101;
  t105 = t27 * t27;
  t108 = t31 * t31;
  t113 = 0.20e0 * t21 * (0.100e1 * t100 * t102 + 0.100e1 * t105 * t102 + 0.100e1 * t108 * t102);
  t117 = 0.10e1 * t39 * (0.10e1 * t72 * t3 + 0.10e1 * (t75 - t89) * t3) + 0.10e1 * t96 * t23 + 0.10e1 * (t99 - t113) * t23;
  t120 = t44 * t44;
  t121 = t46 * t46;
  t122 = 0.1e1 / t121;
  t123 = t120 * t122;
  t125 = t55 * t55;
  t126 = t125 * t122;
  t128 = t63 * t63;
  t129 = t128 * t122;
  t131 = 0.40e0 * t117 * t47 - 0.2000e0 * t123 - 0.2000e0 * t126 - 0.2000e0 * t129;
  t132 = std::sqrt(t131);
  t134 = std::abs(t50 + t58 + t66 + t132);
  t135 = 0.5e0 * t134;
  t137 = std::abs(-t50 - t58 - t66 + t132);
  t138 = 0.5e0 * t137;
  t139 = t135 + t138 - t68;
  t146 = t2 - t22;
  t149 = t8 - t27;
  t152 = t13 - t31;
  t155 = 0.4e0 * (0.500e0 * t123 + 0.500e0 * t126 + 0.500e0 * t129) * t69 - 0.40e0 * t48 * t146 - 0.40e0 * t56 * t149 - 0.40e0 * t64 * t152 + t75 - t99;
  t158 = t139 * t155 / t131;
  t160 = t135 - t138;
  t165 = -t67 * t69 + t146 * t5 + t149 * t10 + t152 * t15;
  t167 = 0.1e1 / t132;
  t168 = t160 * t165 * t167;
  fluxes[0] = 0.5e0 * t1 * t18 + 0.5e0 * t21 * t35 - 0.5e0 * t68 * t69 - 0.5e0 * t158 - 0.5e0 * t168;
  t174 = t75 - t89 + t99 - t113;
  t179 = t158 + t168;
  t186 = t160 * t155 * t167 + t139 * t165;
  fluxes[1] = 0.5e0 * t2 * t18 + 0.5e0 * t22 * t35 + 0.5e0 * t5 * t174 - 0.5e0 * t68 * t146 - 0.50e0 * t179 * t44 * t47 - 0.5e0 * t186 * t5;
  fluxes[2] = 0.5e0 * t8 * t18 + 0.5e0 * t27 * t35 + 0.5e0 * t10 * t174 - 0.5e0 * t68 * t149 - 0.50e0 * t179 * t55 * t47 - 0.5e0 * t186 * t10;
  fluxes[3] = 0.5e0 * t13 * t18 + 0.5e0 * t31 * t35 + 0.5e0 * t15 * t174 - 0.5e0 * t68 * t152 - 0.50e0 * t179 * t63 * t47 - 0.5e0 * t186 * t15;
  fluxes[4] = 0.5e0 * (0.14e1 * t72 - t89) * t18 + 0.5e0 * (0.14e1 * t96 - t113) * t35 - 0.5e0 * t68 * (t72 - t96) - 0.50e0 * t179 * t117 * t47 - 0.5e0 * t186 * t67;
  
}
