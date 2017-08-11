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
  refNodes{},
  dofsF{},
  refNodesF{},
  nQV{},
  xQV{},
  wQV{},
  nQF{},
  xQF{},
  wQF{},
  nStates{},
  Mel{},
  Mipiv{},
  sToS{},
  Kels{},
  KelsF{},
  InterpF{},
  InterpV{},
  InterpW{},
  InterpTk{},
  InterpTkQ{},
  u{},
  p{}
{
  
  // Initialize nodes within elements
  chebyshev2D(order, refNodes);
  dofs = refNodes.size(1)*refNodes.size(2);
  chebyshev1D(order, refNodesF);
  dofsF = refNodesF.size(1);
  
  // Compute interpolation matrices
  precomputeInterpMatrices(); // sets nQ*,xQ*,wQ*,Interp*
  mesh.setupNodes(InterpTk, order);
  mesh.setupQuads(InterpTkQ, nQV);
  mesh.setupJacobians(nQV, xQV, Jk, nQF, xQF, JkF);
  mpi.initDatatype(nQF);
  mpi.initFaces(Mesh::N_FACES);
  
  // Initialize physics
  nStates = Mesh::DIM*(Mesh::DIM+1)/2 + Mesh::DIM;
  initMaterialProps(); // sets mu, lambda, rho
  initTimeStepping(dtSnap); // sets dt, timesteps, stepsPerSnap
  
  // Initialize sources
  initSource(srcParams); // sets p.src
  
  // Initialize u
  // upper diagonal of E in Voigt notation, followed by v
  // u = [e11, e22, e12, v1, v2]
  u.realloc(dofs, nStates, mesh.nElements);
  initialCondition(); // sets u
  
  // Compute local matrices
  precomputeLocalMatrices();
  
  if (mpi.rank == mpi.ROOT) {
    std::cout << "dt = " << dt << std::endl;
    std::cout << "computing for " << timesteps+p.src.halfSrc << " time steps." << std::endl;
    std::cout << "maxvel = " << p.C << std::endl;
  }
  
}

/** Precomputes all the interpolation matrices used by DG method */
void Solver::precomputeInterpMatrices() {
  
  // 1D interpolation matrix for use on faces
  nQF = gaussQuad1D(2*order, xQF, wQF);
  interpolationMatrix1D(refNodesF, xQF, InterpF);
  
  // 2D interpolation matrix for use on elements
  nQV = gaussQuad2D(2*order, xQV, wQV);
  interpolationMatrix2D(refNodes, xQV, InterpV);
  
  // Store weights*InterpV in InterpW to avoid recomputing every time
  InterpW.realloc(nQV, dofs);
  for (int iN = 0; iN < dofs; ++iN) {
    for (int iQ = 0; iQ < nQV; ++iQ) {
      InterpW(iQ, iN) = InterpV(iQ, iN)*wQV(iQ);
    }
  }
  
  // 2D interpolation for use with mappings onto volume nodes
  darray chebyTk;
  chebyshev2D(1, chebyTk);
  interpolationMatrix2D(chebyTk, refNodes, InterpTk);
  
  // 2D interpolation for use with mappings onto volume quadrature pts
  interpolationMatrix2D(chebyTk, xQV, InterpTkQ);
  
}

/** Precomputes all the local matrices used by DG method */
void Solver::precomputeLocalMatrices() {
  
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
    for (int s = 0; s < storage; ++s) {
      
      // localJPI = Jk*Ps*Interp
      switch(s) {
      case 0: {
	// P == I
	for (int iN = 0; iN < dofs; ++iN) {
	  for (int iQ = 0; iQ < nQV; ++iQ) {
	    localJPI(iQ, iN) = Jk(iQ,iK)*InterpV(iQ,iN);
	  }
	}
	break;
      }
      case 1: {
	// P == rho*I
	for (int iN = 0; iN < dofs; ++iN) {
	  for (int iQ = 0; iQ < nQV; ++iQ) {
	    localJPI(iQ, iN) = Jk(iQ,iK)*rhoQ(iQ,iK)*InterpV(iQ,iN);
	  }
	}
	break;
      }
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
  
  // Compute gradient of reference bases on the quadrature points
  darray dPhiQ;
  dPhi2D(refNodes, xQV, dPhiQ);
  
  // Initialize K matrices = dx_l(phi_i)*weights
  Kels.realloc(nQV,dofs,Mesh::DIM);
  for (int l = 0; l < Mesh::DIM; ++l) {
    for (int iN = 0; iN < dofs; ++iN) {
      for (int iQ = 0; iQ < nQV; ++iQ) {
	Kels(iQ, iN, l) = dPhiQ(iQ,iN,l)*wQV(iQ);
      }
    }
  }
  
  // Initialize K matrix for use along faces
  KelsF.realloc(nQF,dofsF);
  for (int iFN = 0; iFN < dofsF; ++iFN) {
    for (int iFQ = 0; iFQ < nQF; ++iFQ) {
      KelsF(iFQ,iFN) = InterpF(iFQ,iFN)*wQF(iFQ);
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

/** Initialize time stepping information using CFL condition */
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
  dt = CFL*mesh.dxMin/(p.C*(std::max(1, order*order)));
  // Ensure we will exactly end at tf
  timesteps = static_cast<dgSize>(std::ceil(tf/dt));
  dt = tf/timesteps;
  // Compute number of time steps per dtSnap sec
  stepsPerSnap = static_cast<dgSize>(std::ceil(dtSnap/dt));
  
}

/** Initialize source */
void Solver::initSource(Source::Params& srcParams) {
  
  srcParams.timesteps = timesteps;
  srcParams.dt = dt;
  
  // Compute vsMin = min(vs) throughout domain
  srcParams.vsMin = p.C;
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int iQ = 0; iQ < nQV; ++iQ) {
      double vsi = std::sqrt(p.mu(iQ,iK)/p.rho(iQ,iK));
      if (vsi < srcParams.vsMin)
	srcParams.vsMin = vsi;
    }
  }
  srcParams.dxMax = mesh.dxMax/std::max(order,1);
  
  p.src.init(srcParams);
  p.src.definePositions(srcParams, mesh);
}

/** Sets u according to an initial condition */
void Solver::initialCondition() {
  
  trueSolution(u, -p.src.halfSrc*dt);
  return;
  
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
  
}

/** Computes the true solution of the plane wave problem at time t in uTrue */
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
    
#ifdef DEBUG
    if (anynan(u)) {
      std::cerr << "ERROR: NaNs in u!" << std::endl;
    }
#endif
    
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
      
      // Output snapshot files
      bool success = initXYZVFile("output/xyu.txt", Mesh::DIM, iTime/stepsPerSnap, "u", nStates);
      if (!success)
	mpi.exit(-1);
      success = exportToXYZVFile("output/xyu.txt", iTime/stepsPerSnap, mesh.globalCoords, u);
      if (!success)
	mpi.exit(-1);
      
      double norm = computeL2Norm(u);
      std::cout << "L2 norm = " << norm << std::endl;
      if (std::isnan(norm)) {
	std::cerr << "ERROR: NaNs detected by norm calculation!" << std::endl;
	mpi.exit(-1);
      }
      
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
  darray onFaces{dofsF, nStates, Mesh::N_FACES, mesh.nElements, dim};
  for (int l = 0; l < dim; ++l) {
    for (int iK = 0; iK < mesh.nElements; ++iK) {
      for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iFN = 0; iFN < dofsF; ++iFN) {
	    onFaces(iFN, iS, iF, iK, l) = curr(mesh.efToN(iFN, iF), iS, iK, l);
	  }
	}
      }
    }
    
    // Face interpolation toInterpF = InterpF*uOnFaces
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		nQF, nStates*Mesh::N_FACES*mesh.nElements, dofsF, 1.0, 
		InterpF.data(), nQF, &onFaces(0,0,0,0,l), dofsF, 
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
    
    // Compute localJWI = Jk*W*Interp
    for (int iN = 0; iN < dofs; ++iN) {
      for (int iQ = 0; iQ < nQV; ++iQ) {
	localJWI(iQ, iN) = Jk(iQ,iK)*InterpW(iQ, iN);
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
  
  // Initialize flux along faces
  darray fStar{nQF, nStates};
  darray faceContributions{dofsF, nStates};
  darray KelsFJ{nQF, dofsF};
  
  darray fluxes{nStates};
  darray uK{nStates};
  darray uN{nStates};
  
  // Loop over all elements
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
      
      // For every face, compute fstar = fc*(uInterpF)
      auto nK = mesh.eToE(iF, iK);
      
      darray normalK{&mesh.normals(0,iF,iK), Mesh::DIM};
      
      if (nK < 0) { // boundary conditions
	
	for (int iFQ = 0; iFQ < nQF; ++iFQ) {
	  
	  // Get all state variables at this point
	  for (int iS = 0; iS < nStates; ++iS) {
	    uK(iS) = uInterpF(iFQ, iS, iF, iK);
	  }
	  // Get material properties at this point
	  double lambdaK = p.lambda(mesh.efToQ(iFQ, iF), iK);
	  double muK = p.mu(mesh.efToQ(iFQ, iF), iK);
	  double rhoK = p.rho(mesh.efToQ(iFQ, iF), iK);
	  
	  // Compute boundary fluxes = F*(u-)'*n
	  boundaryFluxC(uK, normalK, fluxes, static_cast<Mesh::Boundary>(nK), lambdaK, muK, rhoK);
	  // Copy flux data into fStar
	  for (int iS = 0; iS < nStates; ++iS) {
	    fStar(iFQ, iS) = fluxes(iS);
	  }
	  
	}
	
      }
      else {
	
	auto nF = mesh.eToF(iF, iK);
	
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
	
      }
      
      // KelsFJ = KelsF*JkF
      for (int iFN = 0; iFN < dofsF; ++iFN) {
	for (int iFQ = 0; iFQ < nQF; ++iFQ) {
	  KelsFJ(iFQ,iFN) = KelsF(iFQ,iFN)*JkF(iFQ,iF);
	}
      }
      
      // Flux contribution = KelsFJ'*fstar
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
		  dofsF, nStates, nQF, 1.0, KelsFJ.data(), nQF, 
		  fStar.data(), nQF, 
		  0.0, faceContributions.data(), dofsF);
      
      // Add up face contributions into global residual array
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iFN = 0; iFN < dofsF; ++iFN) {
	  residual(mesh.efToN(iFN,iF), iS, iK) += faceContributions(iFN, iS);
	}
      }
      
    }
  }
  
}

/** Evaluates the volume integral term for convection in the RHS */
void Solver::convectDGVolume(const darray& uInterpV, darray& residual) const {
  
  // Contains all flux information
  darray fc{nQV, nStates, Mesh::DIM};
  // JKel = Jk*W*dPhi_l
  darray localJKel{nQV,dofs};
  
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
	  fc(iQ,iS,l) = fluxes(iS,l);
	}
      }
      
    }
    
    for (int l = 0; l < Mesh::DIM; ++l) {
      
      // localJKel = Jk*W*dPhi_l
      for (int iN = 0; iN < dofs; ++iN) {
	for (int iQ = 0; iQ < nQV; ++iQ) {
	  localJKel(iQ,iN) = Jk(iQ,iK)*Kels(iQ,iN,l);
	}
      }
      
      // residual += localJKel_l*fc_l(uInterpV)
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		  dofs, nStates, nQV, 1.0, localJKel.data(), nQV,
		  &fc(0,0,l), nQV, 1.0, &residual(0,0,iK), dofs);
      
    }
  }
  
}


/**
   MPI communication: Start nonblocking Sends/Recvs of pre-interpolated data to other tasks.
   First packs data from interpolated face data into send arrays.
   Assumes interpolated is of size (nQF, nStates, Mesh::N_FACES, nElements+nGElements, dim)
*/
void Solver::mpiStartComm(const darray& interpolated, int dim, darray& toSend, darray& toRecv, MPI_Request * rk4Reqs) const {
  
  if (nStates*max(mesh.mpiNBElems)*dim == 0)
    return; // nothing to communicate
  
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
  
  if (nStates*max(mesh.mpiNBElems)*dim == 0)
    return; // nothing to communicate
  
  // Finalizes sends/recvs
  MPI_Waitall(2*MPIUtil::N_FACES, rk4Reqs, MPI_STATUSES_IGNORE);
  
  // Unpack data
  for (int iF = 0; iF < MPIUtil::N_FACES; ++iF) {
    
    for (int l = 0; l < dim; ++l) {
      for (int bK = 0; bK < mesh.mpiNBElems(iF); ++bK) {
	auto iK = mesh.mpibeToE(bK, iF);
	auto nK = mesh.eToE(mpi.faceMap(iF), iK);
	
	if (nK >= 0) { // not a boundary condition
	  auto nF = mesh.eToF(mpi.faceMap(iF), iK);
	  
	  for (int iS = 0; iS < nStates; ++iS) {
	    for (int iFQ = 0; iFQ < nQF; ++iFQ) {
	      interpolated(iFQ, iS, nF, nK, l) = toRecv(iFQ, iS, bK, l, iF);
	    }
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
  int nBElems = max(mesh.mpiNBElems);
  if (nmaterials*nQF*nBElems == 0)
    return; // nothing to communicate
  
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
      auto nK = mesh.eToE(mpi.faceMap(iF), iK);
      
      if (nK >= 0) { // not a boundary condition
	auto nF = mesh.eToF(mpi.faceMap(iF), iK);
	
	for (int iFQ = 0; iFQ < nQF; ++iFQ) {
	  p.lambda(mesh.efToQ(iFQ,nF),nK) = toRecv(0, iFQ, bK, iF);
	  p.mu(mesh.efToQ(iFQ,nF),nK)     = toRecv(1, iFQ, bK, iF);
	  p.rho(mesh.efToQ(iFQ,nF),nK)    = toRecv(2, iFQ, bK, iF);
	}
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
    
    // JIloc = Jk*Interp
    for (int iN = 0; iN < dofs; ++iN) {
      for (int iQ = 0; iQ < nQV; ++iQ) {
	JIloc(iQ,iN) = Jk(iQ,iK)*InterpV(iQ,iN);
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


//////////////////////////////////////////////////////////////////////////////////
// Functions specific to elastic physics
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
  
}

/**
   Evaluates the convection numerical flux function at a boundary for this PDE 
   for all states, dotted with teh normal, storing output in fluxes.
*/
void Solver::boundaryFluxC(const darray& uK, const darray& normalK, darray& fluxes, 
			   Mesh::Boundary bc, double lambdaK, double muK, double rhoK) const {
  
  int nstrains = Mesh::DIM*(Mesh::DIM+1)/2;
  fluxes.fill(0.0);
  
  switch(bc) {
  case Mesh::Boundary::free: {
    // F(0:2) = -(vxI + Ixv)/2.0 * n
    fluxes(0) = -uK(nstrains+0)*normalK(0);
    fluxes(1) = -uK(nstrains+1)*normalK(1);
    fluxes(2) = -uK(nstrains+1)/2.0*normalK(0) - uK(nstrains+0)/2.0*normalK(1);
    // F(3:4) = 0
    break;
  }
  case Mesh::Boundary::absorbing: {
    double vpK = std::sqrt((lambdaK+2*muK)/rhoK);
    double vsK = std::sqrt(muK/rhoK);
    
    // F(0:2) = -(vxI + Ixv)/2.0 * n
    fluxes(0) = -uK(nstrains+0)*normalK(0);
    fluxes(1) = -uK(nstrains+1)*normalK(1);
    fluxes(2) = -uK(nstrains+1)/2.0*normalK(0) - uK(nstrains+0)/2.0*normalK(1);
    
    // Z = rho(vp nxn + vs (1-nxn))
    double Z00 = rhoK*(vpK*normalK(0)*normalK(0) + vsK*(1-normalK(0)*normalK(0)));
    double Z01 = rhoK*(vpK*normalK(0)*normalK(1) - vsK*(normalK(0)*normalK(1)));
    double Z11 = rhoK*(vpK*normalK(1)*normalK(1) + vsK*(1-normalK(1)*normalK(1)));
    double Z10 = Z01;
    
    // F(3:4) =  Z * v
    fluxes(nstrains+0) = Z00*uK(nstrains+0) + Z01*uK(nstrains+1);
    fluxes(nstrains+1) = Z10*uK(nstrains+0) + Z11*uK(nstrains+1);
    break;
  }
  default: {
    std::cerr << "FATAL ERROR: boundary condition is not a valid choice!" << std::endl;
    mpi.exit(-1);
    break;
  }
  }
  
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
  double C = std::max(vpK, vpN);
  
  fluxes.fill(0.0);
  for (int iS = 0; iS < nStates; ++iS) {
    for (int l = 0; l < Mesh::DIM; ++l) {
      fluxes(iS) += (FN(iS,l) + FK(iS,l))/2.0*normalK(l)
	- C/2.0*(uN(iS) - uK(iS))*normalK(l)*normalK(l);
    }
  }
  
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

