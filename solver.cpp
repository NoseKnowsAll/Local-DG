#include "solver.h"

/** Default constructor */
Solver::Solver() : Solver{2, Source::Params{}, 0.1, 1.0, Mesh{}} { }

/** Main constructor */
Solver::Solver(int _order, Source::Params srcParams, double dtSnap, double _tf, const Mesh& _mesh) :
  mesh{_mesh},
  mpi{_mesh.mpi},
  tf{_tf},
  dt{},
  timesteps{},
  stepsPerSnap{},
  order{_order},
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
  InterpW{},
  u{},
  p{}
{
  
  // Initialize nodes within elements
  chebyshev2D(order, refNodes);
  dofs = refNodes.size(1)*refNodes.size(2);
  mesh.setupNodes(refNodes, order);
  mpi.initDatatype(mesh.nFQNodes);
  mpi.initFaces(Mesh::N_FACES);
  
  // Compute interpolation matrices
  precomputeInterpMatrices(); // sets nQV, xQV
  mesh.setupQuads(xQV, nQV);
  
  // Initialize u and physics
  nStates = Mesh::DIM*(Mesh::DIM+1)/2 + Mesh::DIM;
  // upper diagonal of E in Voigt notation, followed by v
  // u = [e11, e22, e12, v1, v2]
  u.realloc(dofs, nStates, mesh.nElements);
  initMaterialProps(); // sets mu, lambda, rho
  initialCondition(); // sets u
  initTimeStepping(dtSnap); // sets dt, timesteps, stepsPerSnap
  
  // Initialize sources
  srcParams.timesteps = timesteps;
  srcParams.dt = dt;
  p.src.init(srcParams);
  p.src.definePositions(srcParams, mesh, xQV);
  
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
  
  // Compute derivative of reference bases on the quadrature points
  darray xQ, wQ;
  gaussQuad2D(2*order, xQ, wQ);
  darray dPolyQuad;
  dlegendre2D(order, xQ, dPolyQuad);
  darray dPhiQ{nQV,order+1,order+1,Mesh::DIM};
  for (int l = 0; l < Mesh::DIM; ++l) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
		nQV, dofs, dofs, 1.0, &dPolyQuad(0,0,0,l), nQV, 
		coeffsPhi.data(), dofs, 0.0, &dPhiQ(0,0,0,l), nQV);
  }
  
  // Store weights*InterpV in InterpW to avoid recomputing every time
  InterpW.realloc(nQV, dofs);
  for (int iN = 0; iN < dofs; ++iN) {
    for (int iQ = 0; iQ < nQV; ++iQ) {
      InterpW(iQ, iN) = InterpV(iQ, iN)*wQ(iQ);
    }
  }
  
  // Initialize mass matrices = integrate(ps*phi_i*phi_j)
  int storage = 2;
  Mel.realloc(dofs,dofs,storage,mesh.nElements);
  Mipiv.realloc(dofs,storage,mesh.nElements);
  darray localJPI{nQV, dofs};
  darray rhoQ{nQV, mesh.nElements};
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      nQV, mesh.nElements, dofs, 1.0, InterpV.data(), nQV,
	      p.rho.data(), dofs, 0.0, rhoQ.data(), nQV);
  
  // Set state-to-storage mapping
  sToS.realloc(nStates);
  for (int s = 0; s < Mesh::DIM*(Mesh::DIM+1)/2; ++s) {
    sToS(s) = 0; // upper diagonal of strain tensor
  }
  for (int s = Mesh::DIM*(Mesh::DIM+1)/2; s < nStates; ++s) {
    sToS(s) = 1; // velocities
  }
  
  // Compute mass matrix = Interp'*W*Jk*Ps*Interp
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    
    // Compute Jacobian = det(Jacobian(Tk))
    double Jacobian = 1.0;
    for (int l = 0; l < Mesh::DIM; ++l) {
      Jacobian *= mesh.tempMapping(0,l,iK);
    }
      
    for (int s = 0; s < storage; ++s) {
      
      // localJPI = Jk*Ps*Interp
      switch(s) {
      case 0:
	// P == I
	for (int iN = 0; iN < dofs; ++iN) {
	  for (int iQ = 0; iQ < nQV; ++iQ) {
	    localJPI(iQ, iN) = Jacobian*InterpV(iQ, iN);
	  }
	}
	break;
      case 1:
	// P == rho*I
	for (int iN = 0; iN < dofs; ++iN) {
	  for (int iQ = 0; iQ < nQV; ++iQ) {
	    localJPI(iQ, iN) = Jacobian*rhoQ(iQ)*InterpV(iQ, iN);
	  }
	}
	break;
      }
      
      // Mel = Interp'*W*Jk*Ps*Interp
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
		  dofs, dofs, nQV, 1.0, InterpW.data(), nQV, 
		  localJPI.data(), nQV, 0.0, &Mel(0,0,s,iK), dofs);
      
      // Mel overwritten with U*D*U'
      LAPACKE_dsytrf(LAPACK_COL_MAJOR, 'U', dofs,
		     &Mel(0,0,s,iK), dofs, &Mipiv(0,s,iK));
    }
  }
  
  // TODO: everything below here needs to work with new mappings
  darray alpha{Mesh::DIM};
  double Jacobian = 1.0;
  for (int l = 0; l < Mesh::DIM; ++l) {
    alpha(l) = mesh.tempMapping(0,l,0);
    Jacobian *= mesh.tempMapping(0,l,0);
  }
  
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
  
  p.lambda.realloc(nQV, mesh.nElements+mesh.nGElements);
  p.mu.realloc(nQV, mesh.nElements+mesh.nGElements);
  p.rho.realloc(nQV, mesh.nElements+mesh.nGElements);
  
  if (!fileExists) {
    // Initialize material properties to constants
    for (int iK = 0; iK < mesh.nElements; ++iK) {
      for (int iQ = 0; iQ < nQV; ++iQ) {
	p.lambda(iQ,iK) = p.rhoConst*(p.vpConst*p.vpConst - 2*p.vsConst*p.vsConst);
	p.mu(iQ,iK) = p.rhoConst*(p.vsConst*p.vsConst);
	p.rho(iQ,iK) = p.rhoConst;
      }
    }
  }
  else {
    // Interpolate data from input grid onto quadrature points
    for (int iK = 0; iK < mesh.nElements; ++iK) {
      for (int iQ = 0; iQ < nQV; ++iQ) {
	darray coord{&mesh.globalQuads(0,iQ,iK), Mesh::DIM};
	
	double vpi = gridInterp(coord, vp, origins, deltas);
	double vsi = gridInterp(coord, vs, origins, deltas);
	double rhoi = gridInterp(coord, rhoIn, origins, deltas);
	
	// Isotropic elastic media formula
	p.lambda(iQ,iK) = rhoi*(vpi*vpi - 2*vsi*vsi);
	p.mu(iQ,iK) = rhoi*(vsi*vsi);
	p.rho(iQ,iK) = rhoi;
	
      }
    }
  }
  
  // Send material information on MPI boundaries to neighbors
  mpiSendMaterials();
  
}

/* Initialize time stepping information using CFL condition */
void Solver::initTimeStepping(double dtSnap) {
  
  // Compute p.C = maxvel = max(vp) throughout domain
  p.C = 0.0;
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int iQ = 0; iQ < nQV; ++iQ) {
      double vpi = std::sqrt((p.lambda(iQ,iK) + 2*p.mu(iQ,iK))/p.rho(iQ,iK));
      if (vpi > p.C)
	p.C = vpi;
    }
  }
  
  // Choose dt based on CFL condition
  double CFL = 0.1;
  dt = CFL*std::min(mesh.minDX, mesh.minDY)/(p.C*(std::max(1, order*order)));
  // Ensure we will exactly end at tf
  timesteps = static_cast<dgSize>(std::ceil(tf/dt));
  dt = tf/timesteps;
  // Compute number of time steps per dtSnap sec
  stepsPerSnap = static_cast<dgSize>(std::ceil(dtSnap/dt));
  
}

/** Sets u according to an initial condition */
void Solver::initialCondition() {
  
  trueSolution(u, -p.src.halfSrc*dt);
  
  /*
  // Sin function allowing for periodic initial condition
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int iN = 0; iN < dofs; ++iN) {
      
      double x = mesh.globalCoords(0,iN,iK);
      double y = mesh.globalCoords(1,iN,iK);
      
      u(iN,0,iK) = 0.0;
      u(iN,1,iK) = 0.0;
      u(iN,2,iK) = 0.0;
      u(iN,3,iK) = 0.0;
      u(iN,4,iK) = 0.0;
      
    }
  }
  */
  
}

/** Computes the true solution at time t in uTrue */
void Solver::trueSolution(darray& uTrue, double t) const {
  
  double pd[Mesh::DIM] = {1, 0};
  double dp[Mesh::DIM] = {1, 0};
  double ds[Mesh::DIM] = {0, 1};
  double k = 2*M_PI;
  
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int iN = 0; iN < dofs; ++iN) {
      
      double x[Mesh::DIM] = {mesh.globalCoords(0,iN,iK), mesh.globalCoords(1,iN,iK)};
      
      double pval = k*(x[0]*pd[0]+x[1]*pd[1]-p.vpConst*t);
      double sval = k*(x[0]*pd[0]+x[1]*pd[1]-p.vsConst*t);
      
      // E
      uTrue(iN, 0, iK) = -k*(pd[0]*dp[0]*std::sin(pval) + pd[0]*ds[0]*std::sin(sval));
      uTrue(iN, 1, iK) = -k*(pd[1]*dp[1]*std::sin(pval) + pd[1]*ds[1]*std::sin(sval));
      uTrue(iN, 2, iK) = -k*(pd[0]*dp[1]*std::sin(pval) + pd[0]*ds[1]*std::sin(sval)
			   + pd[1]*dp[0]*std::sin(pval) + pd[1]*ds[0]*std::sin(sval))/2.0;
      // v
      uTrue(iN, 3, iK) = k*(p.vpConst*dp[0]*std::sin(pval) + p.vsConst*ds[0]*std::sin(sval));
      uTrue(iN, 4, iK) = k*(p.vpConst*dp[1]*std::sin(pval) + p.vsConst*ds[1]*std::sin(sval));
      
    }
  }
}

/**
   Actually time step DG method according to RK4, 
   updating the solution u every time step
*/
void Solver::dgTimeStep() {
  
  // Allocate working memory
  int nQF = mesh.nFQNodes;
  darray uCurr{dofs, nStates, mesh.nElements, 1};
  darray uInterpF{nQF, nStates, Mesh::N_FACES, mesh.nElements+mesh.nGElements, 1};
  darray uInterpV{nQV, nStates, mesh.nElements, 1};
  darray ks{dofs, nStates, mesh.nElements, rk4::nStages};
  darray uTrue{dofs, nStates, mesh.nElements, 1};
  darray pressure{nQV, 1, mesh.nElements};
  
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
  for (int iTime = -p.src.halfSrc; iTime < timesteps; ++iTime) {
    
    if (mpi.rank == mpi.ROOT) {
      std::cout << "time = " << iTime*dt << std::endl;
    }
    
    if (iTime % stepsPerSnap == 0 && iTime >= 0) {
      if (mpi.rank == mpi.ROOT) {
	auto endTime = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = endTime-startTime;
	std::cout << "Saving snapshot " << iTime/stepsPerSnap << "...\n";
	std::cout << "Elapsed time so far = " << elapsed.count() << std::endl;
      }
      
      // Compute true solution
      trueSolution(uTrue, iTime*dt);
      
      // Output snapshot files
      bool success = initXYZVFile("output/xyu.txt", Mesh::DIM, iTime/stepsPerSnap, "u", nStates);
      if (!success)
	exit(-1);
      success = exportToXYZVFile("output/xyu.txt", iTime/stepsPerSnap, mesh.globalCoords, u);
      if (!success)
	exit(-1);
      
      // Compute error
      double error = computeL2Error(u, uTrue);
      std::cout << "L2 error = " << error << std::endl;
      
      // TODO: debugging
      trueSolution(uTrue, iTime*dt);
      error = computeInfError(u, uTrue);
      std::cout << "inf error = " << error << std::endl;
      
    }
    
    // Use RK4 to compute k values at each stage
    for (int istage = 0; istage < rk4::nStages; ++istage) {
      
      // Updates uCurr = u+dt*a(s,s-1)*ks(:,s)
      rk4::updateCurr(uCurr, u, ks, dt, istage);
      
      // Updates ks(:,istage) = rhs(uCurr) based on DG method
      rk4Rhs(uCurr, uInterpF, uInterpV, 
	     toSend, toRecv, rk4Reqs, 
	     ks, istage, iTime);
      
    }
    
    // Use RK4 to move to next time step
    rk4::integrateTime(u, ks, dt);
    
  }
  
  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = endTime-startTime;
  if (mpi.rank == mpi.ROOT) {
    std::cout << "Finished time stepping. Time elapsed = " << elapsed.count() << std::endl;
  }
  
}


/**
   Computes the RHS of Runge Kutta method for use with DG method.
   Updates ks(:,istage) with rhs evaluated at uCurr
*/
void Solver::rk4Rhs(const darray& uCurr, darray& uInterpF, darray& uInterpV, 
		    darray& toSend, darray& toRecv, MPI_Request * rk4Reqs,
		    darray& ks, int istage, int iTime) const {
  
  // Interpolate uCurr once
  interpolate(uCurr, uInterpF, uInterpV, toSend, toRecv, rk4Reqs, 1);
  
  // Now compute ks(:,istage) from uInterp according to:
  // ks(:,istage) = Mel\( B + K(u) - F(u) )
  
  darray residual{&ks(0,0,0,istage), dofs, nStates, mesh.nElements};
  residual.fill(0.0);
  
  // ks(:,istage) += -F(u)
  convectDGFlux(uInterpF, residual);
  cblas_dscal(dofs*nStates*mesh.nElements, -1.0, residual.data(), 1);
  
  // ks(:,istage) += K(u)
  convectDGVolume(uInterpV, residual);
  
  // ks(:,istage) += B
  sourceVolume(residual, istage, iTime);
  
  // ks(:,istage) = Mel\ks(:,istage)
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int s = 0; s < nStates; ++s) {
      LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'U', dofs, 1,
		     &Mel(0,0,sToS(s),iK), dofs, &Mipiv(0,sToS(s),iK),
		     &residual(0,s,iK), dofs);
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
   Computes right-hand-side B term for use in the DG formulation
*/
void Solver::sourceVolume(darray& residual, int istage, int iTime) const {
  
  darray localJWI{nQV, dofs};
  darray f{nQV};
  darray b{dofs};
  auto waveAmp = p.src.wavelet(istage, iTime+p.src.halfSrc);
  
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    
    // Compute Jacobian = det(Jacobian(Tk))
    double Jacobian = 1.0;
    for (int l = 0; l < Mesh::DIM; ++l) {
      Jacobian *= mesh.tempMapping(0,l,iK);
    }
    
    // Compute localJWI = J*W*Interp
    for (int iN = 0; iN < dofs; ++iN) {
      for (int iQ = 0; iQ < nQV; ++iQ) {
	localJWI(iQ, iN) = Jacobian*InterpW(iQ, iN);
      }
    }
    
    // Compute f = rho(x)*g(x)*w(t)
    for (int iQ = 0; iQ < nQV; ++iQ) {
      f(iQ) = p.rho(iQ,iK)*p.src.weights(iQ,iK)*waveAmp;
    }
    
    // Compute b = Interp'*W*J*f
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		dofs, 1, nQV, 1.0, localJWI.data(), nQV,
		f.data(), nQV, 0.0, b.data(), dofs);
    
    // Add into global residual array
    for (int iS = 0; iS < nStates; ++iS) {
      if (sToS(iS) == 0) {
	// residual += 0
      }
      else if (sToS(iS) == 1) {
	// residual += b
	for (int iN = 0; iN < dofs; ++iN) {
	  residual(iN,iS,iK) += b(iN);
	}
      }
    }
    
  }
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
      
      darray normalK{&mesh.normals(0,iF,iK), Mesh::DIM};
      
      for (int iFQ = 0; iFQ < nQF; ++iFQ) {
	
	// Get all state variables at this point
	for (int iS = 0; iS < nStates; ++iS) {
	  uK(iS) = uInterpF(iFQ, iS, iF, iK);
	  uN(iS) = uInterpF(iFQ, iS, nF, nK);
	}
	// Get material properties at this point
	double lambdaK = p.lambda(mesh.efToQ(iFQ, iF), iK);
	double lambdaN = p.lambda(mesh.efToQ(iFQ, nF), nK);
	double muK = p.mu(mesh.efToQ(iFQ, iF), iK);
	double muN = p.mu(mesh.efToQ(iFQ, nF), nK);
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
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int iQ = 0; iQ < nQV; ++iQ) {
      
      // Copy data into uK
      for (int iS = 0; iS < nStates; ++iS) {
	uK(iS) = uInterpV(iQ,iS,iK);
      }
      // Compute fluxes
      fluxC(uK, fluxes, p.lambda(iQ,iK), p.mu(iQ,iK));
      // Copy data into fc
      for (int l = 0; l < Mesh::DIM; ++l) {
	for (int iS = 0; iS < nStates; ++iS) {
	  fc(iQ,iS,iK,l) = fluxes(iS, l);
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
	  for (int iFQ = 0; iFQ < nQF; ++iFQ) {
	    toSend(iFQ, iS, bK, l, iF) = interpolated(iFQ, iS, mpi.faceMap(iF), iK, l);
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
  MPI_Waitall(2*MPIUtil::N_FACES, rk4Reqs, MPI_STATUSES_IGNORE);
  
  // Unpack data
  for (int iF = 0; iF < MPIUtil::N_FACES; ++iF) {
    
    for (int l = 0; l < dim; ++l) {
      for (int bK = 0; bK < mesh.mpiNBElems(iF); ++bK) {
	auto iK = mesh.mpibeToE(bK, iF);
	auto nF = mesh.eToF(mpi.faceMap(iF), iK);
	auto nK = mesh.eToE(mpi.faceMap(iF), iK);
	
	
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iFQ = 0; iFQ < nQF; ++iFQ) {
	    interpolated(iFQ, iS, nF, nK, l) = toRecv(iFQ, iS, bK, l, iF);
	  }
	}
      }
    }
    
  }
  
}



/**
   MPI communication: Resolves material properties so that all tasks also get 
   their neighbor's material properties on the quadrature points of boundary.
   Note that this is not the same as getting the boundary quadrature points,
   but material properties are linearly interpolated to begin with...
   Good enough and saves a lot of memory.
*/
void Solver::mpiSendMaterials() {
  
  int nmaterials = 3;
  int nQF = mesh.nFQNodes;
  int nBElems = max(mesh.mpiNBElems);
  darray toSend{nmaterials, nQF, nBElems, MPIUtil::N_FACES};
  darray toRecv{nmaterials, nQF, nBElems, MPIUtil::N_FACES};
  // Requests for MPI: 2*face == send, 2*face+1 == recv
  MPI_Request reqs[2*MPIUtil::N_FACES];
  
  // Pack data to send
  for (int iF = 0; iF < MPIUtil::N_FACES; ++iF) {
    for (int bK = 0; bK < mesh.mpiNBElems(iF); ++bK) {
      int iK = mesh.mpibeToE(bK, iF);
      
      for (int iFQ = 0; iFQ < nQF; ++iFQ) {
	toSend(0, iFQ, bK, iF) = p.lambda(mesh.efToQ(iFQ, mpi.faceMap(iF)), iK);
	toSend(1, iFQ, bK, iF) = p.mu(    mesh.efToQ(iFQ, mpi.faceMap(iF)), iK);
	toSend(2, iFQ, bK, iF) = p.rho(   mesh.efToQ(iFQ, mpi.faceMap(iF)), iK);
      }
    }
  }
  
  // Actually send/recv the data
  for (int iF = 0; iF < MPIUtil::N_FACES; ++iF) {
    
    MPI_Isend(&toSend(0,0,0,iF), nmaterials*nQF*mesh.mpiNBElems(iF),
	      MPI_DOUBLE, mpi.neighbors(iF), iF,
	      mpi.cartComm, &reqs[2*iF]);
    
    MPI_Irecv(&toRecv(0,0,0,iF), nmaterials*nQF*mesh.mpiNBElems(iF),
	      MPI_DOUBLE, mpi.neighbors(iF), mpi.tags(iF),
	      mpi.cartComm, &reqs[2*iF+1]);
  }
  
  // Finalize sends/recv
  MPI_Waitall(2*MPIUtil::N_FACES, reqs, MPI_STATUSES_IGNORE);
  
  // Unpack data
  for (int iF = 0; iF < MPIUtil::N_FACES; ++iF) {
    for (int bK = 0; bK < mesh.mpiNBElems(iF); ++bK) {
      auto iK = mesh.mpibeToE(bK, iF);
      auto nF = mesh.eToF(mpi.faceMap(iF), iK);
      auto nK = mesh.eToE(mpi.faceMap(iF), iK);
      
      for (int iFQ = 0; iFQ < nQF; ++iFQ) {
	p.lambda(mesh.efToQ(iFQ,nF),nK) = toRecv(0, iFQ, bK, iF);
	p.mu(mesh.efToQ(iFQ,nF),nK)     = toRecv(1, iFQ, bK, iF);
	p.rho(mesh.efToQ(iFQ,nF),nK)    = toRecv(2, iFQ, bK, iF);
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
  fluxes(nstrains+0,1) = -(2*mu)*uK(2);
  fluxes(nstrains+1,0) = -(2*mu)*uK(2);
  fluxes(nstrains+1,1) = -(lambda+2*mu)*uK(1) - lambda*uK(0);
  
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
  double vpK = std::sqrt((lambdaK + 2*muK)/rhoK);
  double vpN = std::sqrt((lambdaN + 2*muN)/rhoN);
  //double C = (vpK+vpN)/2.0;
  double C = std::max(vpK, vpN); // TODO: is this better?
  
  fluxes.fill(0.0);
  for (int iS = 0; iS < nStates; ++iS) {
    for (int l = 0; l < Mesh::DIM; ++l) {
      fluxes(iS) += (FN(iS,l) + FK(iS,l))/2.0*normalK(l)
	- C/2.0*(uN(iS) - uK(iS))*normalK(l)*normalK(l);
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

/**
   Computes the pressure based off of the interpolated input wavefield uInterpV
   Assumes hydrostatic pressure is the only pressure
*/
void Solver::computePressure(const darray& uInterpV, darray& pressure) const {
  
  // pressure = average of normal stresses = 1/d*sum(S_{ll})
  
  // Computed at quadrature points because we only have 
  // material properties at quadrature points
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int iQ = 0; iQ < nQV; ++iQ) {
      
      pressure(iQ,0,iK) = 0.0;
      for (int iS = 0; iS < Mesh::DIM; ++iS) {
	pressure(iQ,0,iK) += (p.lambda(iQ,iK) + 2.0*p.mu(iQ,iK)/Mesh::DIM)*uInterpV(iQ,iS,iK);
      }
      
    }
  }
  
}


/**
   Compute the L2 error ||u-uTrue||_{L^2} of a function defined at the nodes.
   Reuses the storage in uTrue to compute this error.
*/
double Solver::computeL2Error(const darray& uCurr, darray& uTrue) const {
  
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int iS = 0; iS < nStates; ++iS) {
      for (int iN = 0; iN < dofs; ++iN) {
	uTrue(iN,iS,iK) -= uCurr(iN,iS,iK);
      }
    }
  }
  
  return computeL2Norm(uTrue);
}

/**
   Compute the L2 norm ||u||_{L^2} of a function defined at the nodes.
   Norm = u'*Mel*u
*/
double Solver::computeL2Norm(const darray& uCurr) const {
  
  darray Mloc{dofs, dofs};
  darray JIloc{nQV, dofs};
  darray Mu{dofs,nStates};
  darray norms{nStates, nStates}; // diagonal contains L2Norm of each state
  darray l2Norms{nStates};
  
  for (int iS = 0; iS < nStates; ++iS) {
    l2Norms(iS) = 0.0;
  }
  
  // Add local contribution to norm at each element
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    
    // Compute Jacobian = det(Jacobian(Tk))
    double Jacobian = 1.0;
    for (int l = 0; l < Mesh::DIM; ++l) {
      Jacobian *= mesh.tempMapping(0,l,iK);
    }
    
    // JIloc = Jk*Interp
    for (int iN = 0; iN < dofs; ++iN) {
      for (int iQ = 0; iQ < nQV; ++iQ) {
	JIloc(iQ,iN) = Jacobian*InterpV(iQ,iN);
      }
    }
    
    // Mloc = Interp'*W*Jk*Interp
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		dofs, dofs, nQV, 1.0, InterpW.data(), nQV,
		JIloc.data(), nQV, 0.0, Mloc.data(), dofs);
    
    // Mu = Mloc*u
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		dofs, nStates, dofs, 1.0, Mloc.data(), dofs,
		&uCurr(0,0,iK), dofs, 0.0, Mu.data(), dofs);
    
    // L2Norm = Mu'*u
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		nStates, nStates, dofs, 1.0, Mu.data(), dofs,
		&uCurr(0,0,iK), dofs, 0.0, norms.data(), nStates);
    
    // Add local contribution to norm
    for (int iS = 0; iS < nStates; ++iS) {
      l2Norms(iS) += norms(iS,iS);
    }
    
  }
  
  return max(l2Norms);
  
}

/**
   Compute the Inf error ||u-uTrue||_{\infty} of a function defined at the nodes.
   Reuses the storage in uTrue to compute this error.
*/
double Solver::computeInfError(const darray& uCurr, darray& uTrue) const {
  
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int iS = 0; iS < nStates; ++iS) {
      for (int iN = 0; iN < dofs; ++iN) {
	uTrue(iN,iS,iK) -= uCurr(iN,iS,iK);
      }
    }
  }
  
  return computeInfNorm(uTrue);
  
}

/**
   Compute the infinity-norm ||u||_{\infty} of a function defined at the nodes.
   Norm = max(abs(u))
*/
double Solver::computeInfNorm(const darray& uCurr) const {
  return infnorm(uCurr);
}
