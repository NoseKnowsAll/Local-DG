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
  Mipiv{},
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
  //double maxVel = *std::max_element(a, a+Mesh::DIM);
  double maxVel = std::accumulate(a, a+Mesh::DIM, 0.0);
  dt = 0.1*std::min(std::min(mesh.minDX, mesh.minDY), mesh.minDZ)/(maxVel*(2*order+1));
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
  int nQ2D   = gaussQuad2D(2*order, xQ2D, wQ2D);
  
  interpolationMatrix2D(cheby2D, xQ2D, Interp2D);
  
  // 3D interpolation matrix for use on elements
  darray cheby3D;
  chebyshev3D(order, cheby3D);
  darray xQ3D, wQ3D;
  int nQ3D   = gaussQuad3D(2*order, xQ3D, wQ3D);
  
  interpolationMatrix3D(cheby3D, xQ3D, Interp3D);
  
  /** TODO: Debugging interpolation matrix *
  bool success = initXYZVFile("output/xyzu.txt", "u");
  if (!success)
    exit(-1);
  success = exportToXYZVFile("output/xyzu.txt", mesh.globalCoords, u);
  if (!success)
    exit(-1);
  
  darray uInterp{nQ3D, nStates, mesh.nElements};
  darray uInterp2D{Interp2D.size(0), nStates, Mesh::N_FACES, mesh.nElements};
  interpolateU(u, uInterp2D, uInterp);
  
  // Scales and translates quadrature nodes into each element
  darray globalQuads{Mesh::DIM, nQ3D, mesh.nElements};
  for (int k = 0; k < mesh.nElements; ++k) {
    darray botLeft{&mesh.vertices(0, mesh.eToV(0, k)), Mesh::DIM};
    darray topRight{&mesh.vertices(0, mesh.eToV(7, k)), Mesh::DIM};
    
    for (int iN = 0; iN < nQ3D; ++iN) {
      for (int l = 0; l < Mesh::DIM; ++l) {
	// amount in [0,1] to scale lth dimension
	double scale = .5*(xQ3D(l,iN)+1.0);
	globalQuads(l,iN,k) = botLeft(l)+scale*(topRight(l)-botLeft(l));
      }
    }
  }
  
  success = initXYZVFile("output/xyzuInterp.txt", "uInterp3D");
  if (!success)
    exit(-1);
  success = exportToXYZVFile("output/xyzuInterp.txt", globalQuads, uInterp);
  if (!success)
    exit(-1);
  
  // Scales and translates quadrature nodes into each element
  int size1D = (int)std::ceil(order+1/2.0);
  darray globalQuads2D{Mesh::DIM, size1D, size1D, Mesh::N_FACES, mesh.nElements};
  for (int k = 0; k < mesh.nElements; ++k) {
    darray botLeft{&mesh.vertices(0, mesh.eToV(0, k)), Mesh::DIM};
    darray topRight{&mesh.vertices(0, mesh.eToV(7, k)), Mesh::DIM};
    
    // -x direction face
    for (int iz = 0; iz < size1D; ++iz) {
      for (int iy = 0; iy < size1D; ++iy) {
	globalQuads2D(0,iy,iz, 0,k) = -1.0;
	globalQuads2D(1,iy,iz, 0,k) = xQ2D(0,iy,iz);
	globalQuads2D(2,iy,iz, 0,k) = xQ2D(1,iy,iz);
      }
    }
  
    // +x direction face
    for (int iz = 0; iz < size1D; ++iz) {
      for (int iy = 0; iy < size1D; ++iy) {
	globalQuads2D(0,iy,iz, 1,k) = 1.0;
	globalQuads2D(1,iy,iz, 1,k) = xQ2D(0,iy,iz);
	globalQuads2D(2,iy,iz, 1,k) = xQ2D(1,iy,iz);
      }
    }
    
    // -y direction face
    for (int iz = 0; iz < size1D; ++iz) {
      for (int ix = 0; ix < size1D; ++ix) {
	globalQuads2D(0,ix,iz, 2,k) = xQ2D(0,ix,iz);
	globalQuads2D(1,ix,iz, 2,k) = -1.0;
	globalQuads2D(2,ix,iz, 2,k) = xQ2D(1,ix,iz);
      }
    }
    
    // +y direction face
    for (int iz = 0; iz < size1D; ++iz) {
      for (int ix = 0; ix < size1D; ++ix) {
	globalQuads2D(0,ix,iz, 3,k) = xQ2D(0,ix,iz);
	globalQuads2D(1,ix,iz, 3,k) = 1.0;
	globalQuads2D(2,ix,iz, 3,k) = xQ2D(1,ix,iz);
      }
    }
    
    // -z direction face
    for (int iy = 0; iy < size1D; ++iy) {
      for (int ix = 0; ix < size1D; ++ix) {
	globalQuads2D(0,ix,iy, 4,k) = xQ2D(0,ix,iy);
	globalQuads2D(1,ix,iy, 4,k) = xQ2D(1,ix,iy);
	globalQuads2D(2,ix,iy, 4,k) = -1.0;
      }
    }
    
    // +z direction face
    for (int iy = 0; iy < size1D; ++iy) {
      for (int ix = 0; ix < size1D; ++ix) {
	globalQuads2D(0,ix,iy, 5,k) = xQ2D(0,ix,iy);
	globalQuads2D(1,ix,iy, 5,k) = xQ2D(1,ix,iy);
	globalQuads2D(2,ix,iy, 5,k) = 1.0;
      }
    }
    
    for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
      for (int iy = 0; iy < size1D; ++iy) {
	for (int ix = 0; ix < size1D; ++ix) {
	  for (int l = 0; l < Mesh::DIM; ++l) {
	    // amount in [0,1] to scale lth dimension
	    double scale = .5*(globalQuads2D(l,ix,iy,iF,k)+1.0);
	    globalQuads2D(l,ix,iy,iF,k) = botLeft(l)+scale*(topRight(l)-botLeft(l));
	  }
	}
      }
    }
  }
  
  success = initXYZVFile("output/xyzuInterp2D.txt", "uInterp2D");
  if (!success)
    exit(-1);
  success = exportToXYZVFile("output/xyzuInterp2D.txt", globalQuads2D, uInterp2D);
  if (!success)
    exit(-1);
  
  
  exit(0);
  /** END TODO */
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
  
  //std::cout << "Kels = [" << std::endl;
  //std::cout << Kels << std::endl;
  //std::cout << "];" << std::endl;
  //std::cout << "Kels = reshape(Mel, [" << dofs << ", " << dofs << "]);" << std::endl;
  
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
  
  M2D.realloc(fDofs,fDofs,Mesh::DIM);
  for (int l = 0; l < Mesh::DIM; ++l) {
    double scaleL = Jacobian/alpha(l);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		fDofs,fDofs,fSizeQ, scaleL, Kels2D.data(), fSizeQ,
		Interp2D.data(), fSizeQ, 0.0, &M2D(0,0,l), fDofs);
  }
  
}

// TODO: Initialize u0 based off of a reasonable function
void Solver::initialCondition() {
  
  // Gaussian distribution with variance 0.2, centered around 0.5
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
	    
	    u(vID, iS, k) = a*std::exp((-std::pow(x-.5,2.0)-std::pow(y-.5,2.0)-std::pow(z-.5,2.0))/(2*sigmaSq));
	    
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
  
  // Allocate working memory
  darray uCurr{dofs, nStates, mesh.nElements};
  darray uInterp2D{Interp2D.size(0), nStates, Mesh::N_FACES, mesh.nElements};
  darray uInterp3D{Interp3D.size(0), nStates,  mesh.nElements};
  darray ks{dofs, nStates, mesh.nElements, nStages};
  darray Dus{dofs, nStates, mesh.nElements, Mesh::DIM};
  darray DuInterp2D{Interp2D.size(0), nStates, Mesh::N_FACES, mesh.nElements, Mesh::DIM};
  darray DuInterp3D{Interp3D.size(0), nStates,  mesh.nElements, Mesh::DIM};
  
  std::cout << "Time stepping until tf = " << tf << std::endl;
  
  auto startTime = std::chrono::high_resolution_clock::now();
  
  // Loop over time steps
  for (int iStep = 0; iStep < timesteps; ++iStep) {
    std::cout << "time = " << iStep*dt << std::endl;
    
    if (iStep % 10 == 0) {
      std::cout << "Saving snapshot " << iStep/10 << "...\n";
      bool success = initXYZVFile("output/xyzu.txt", iStep/10, "u");
      if (!success)
	exit(-1);
      success = exportToXYZVFile("output/xyzu.txt", iStep/10, mesh.globalCoords, u);
      if (!success)
	exit(-1);
      
      if (iStep/10 == 10) {
	std::cout << "exiting for debugging purposes...\n";
	exit(0);
      }
      
    }
    
    // Use RK4 to compute k values at each stage
    for (int istage = 0; istage < nStages; ++istage) {
      
      // Updates uCurr = u+dt*a(s,s-1)*ks(:,s)
      rk4UpdateCurr(uCurr, diagA, ks, istage);
      
      // Updates ks(:,istage) = rhs(uCurr) based on DG method
      rk4Rhs(uCurr, uInterp2D, uInterp3D, Dus, DuInterp2D, DuInterp3D, ks, istage);
      
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
void Solver::rk4Rhs(const darray& uCurr, darray& uInterp2D, darray& uInterp3D, 
		    darray& Dus, darray& DuInterp2D, darray& DuInterp3D, 
		    darray& ks, int istage) const {
  
  // Interpolate uCurr once
  interpolateU(uCurr, uInterp2D, uInterp3D);
  
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
  interpolateDus(Dus, DuInterp2D, DuInterp3D);
  
  // Now compute ks(:,istage) from uCurr and these Dus according to:
  // ks(:,istage) = Mel\( K*fc(u) + K*fv(u,Dus) - Fc(u) - Fv(u,Dus) )
  
  darray residual{&ks(0,0,0,istage), dofs, nStates, mesh.nElements};
  residual.fill(0.0);
  
  convectDGFlux(uInterp2D, residual);
  // TODO: ensure this works 
  viscousDGFlux(uInterp2D, DuInterp2D, residual);
  
  // ks(:,istage) = -Fc(u)-Fv(u,Dus)
  cblas_dscal(dofs*nStates*mesh.nElements, -1.0, residual.data(), 1);
  
  // ks(:,istage) += Kc*fc(u)
  convectDGVolume(uInterp3D, residual);
  
  // ks(:,istage) += Kv*fv(u)
  // TODO: ensure this works
  viscousDGVolume(uInterp3D, DuInterp3D, residual);
  
  //std::cout << "res = [" << std::endl;
  //std::cout << residual << std::endl;
  //std::cout << "];" << std::endl;
  //std::cout << "res = reshape(res, [" << dofs << ", " << nStates*mesh.nElements << "]);" << std::endl;
  
  // ks(:,istage) = Mel\ks(:,istage)
  int info = LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'U', dofs, nStates*mesh.nElements,
			    Mel.data(), dofs, Mipiv.data(), residual.data(), dofs);
  
  //std::cout << "answer = [" << std::endl;
  //std::cout << residual << std::endl;
  //std::cout << "];" << std::endl;
  //std::cout << "answer = reshape(answer, [" << dofs << ", " << nStates*mesh.nElements << "]);" << std::endl;
  
  //exit(0);
  
}

/**
   Interpolates u on faces to 2D quadrature points and stores in uInterp2D.
   Interpolates u onto 3D quadrature points and stores in uInterp3D.
*/
void Solver::interpolateU(const darray& uCurr, darray& uInterp2D, darray& uInterp3D) const {
  
  // First grab u on faces and pack into array uOnFaces
  int nFN = mesh.nFNodes;
  int nQ2D = mesh.nFQNodes;
  darray uOnFaces{nFN, nStates, Mesh::N_FACES, mesh.nElements};
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iFN = 0; iFN < nFN; ++iFN) {
	  uOnFaces(iFN, iS, iF, iK) = uCurr(mesh.efToN(iFN, iF), iS, iK);
	}
      }
    }
  }
  // 2D interpolation uInterp2D = Interp2D*uOnFaces
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      nQ2D, nStates*Mesh::N_FACES*mesh.nElements, nFN, 1.0, 
	      Interp2D.data(), nQ2D, uOnFaces.data(), nFN, 
	      0.0, uInterp2D.data(), nQ2D);
  
  // 3D interpolation uInterp3D = Interp3D*uCurr
  int nQ3D = Interp3D.size(0);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      nQ3D, nStates*mesh.nElements, dofs, 1.0, Interp3D.data(), dofs,
	      uCurr.data(), dofs, 0.0, uInterp3D.data(), nQ3D);
  
}

/**
   Interpolates Dus on faces to 2D quadrature points and stores in DuInterp2D.
   Interpolates Dus onto 3D quadrature points and stores in DuInterp3D.
*/
void Solver::interpolateDus(const darray& Dus, darray& DuInterp2D, darray& DuInterp3D) const {
  
  // First grab u on faces and pack into array uOnFaces
  int nFN = mesh.nFNodes;
  int nQ2D = mesh.nFQNodes;
  darray DuOnFaces{nFN, nStates, Mesh::N_FACES, mesh.nElements, Mesh::DIM};
  for (int l = 0; l < Mesh::DIM; ++l) {
    for (int iK = 0; iK < mesh.nElements; ++iK) {
      for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iFN = 0; iFN < nFN; ++iFN) {
	    DuOnFaces(iFN, iS, iF, iK, l) = Dus(mesh.efToN(iFN, iF), iS, iK, l);
	  }
	}
      }
    }
  }
  // 2D interpolation DuInterp2D = Interp2D*DuOnFaces
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      nQ2D, nStates*Mesh::N_FACES*mesh.nElements*Mesh::DIM, nFN, 1.0,
	      Interp2D.data(), nQ2D, DuOnFaces.data(), nFN, 
	      0.0, DuInterp2D.data(), nQ2D);
  
  // 3D interpolation DuInterp3D = Interp3D*Dus
  int nQ3D = Interp3D.size(0);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	      nQ3D, nStates*mesh.nElements*Mesh::DIM, dofs, 1.0, Interp3D.data(), dofs,
	      Dus.data(), dofs, 0.0, DuInterp3D.data(), nQ3D);
  
}


/**
   Local DG Flux: Computes Fl(u) for use in the local DG formulation of second-order terms.
   Uses a downwind formulation for the flux term. 
   Updates residuals arrays with added flux.
*/
void Solver::localDGFlux(const darray& uInterp2D, darray& residuals) const {
  
  int nFN = mesh.nFNodes;
  int nQ2D = mesh.nFQNodes;
  
  // Loop over all elements
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    
    // Initialize flux along faces
    darray fStar{nQ2D, nStates, Mesh::N_FACES, Mesh::DIM}; // TODO: should we move this out of for loop
    darray faceContributions{nFN, nStates, Mesh::N_FACES, Mesh::DIM};
    
    // There are l equations to handle in this flux term
    for (int l = 0; l < Mesh::DIM; ++l) {
      
      // For every face, compute fstar = fl*(Interp2D*u)
      for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
	
	auto nF = mesh.eToF(iF, iK);
	auto nK = mesh.eToE(iF, iK);
	
	auto normalK = mesh.normals(l, iF, iK);
	auto normalN = mesh.normals(l, nF, nK);
	
	// Must compute nStates of these flux integrals per face
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iFQ = 0; iFQ < nQ2D; ++iFQ) {
	    auto uK = uInterp2D(iFQ, iF, iS, iK);
	    auto uN = uInterp2D(iFQ, nF, iS, nK);
	    
	    fStar(iFQ, iS, iF, l) = numericalFluxL(uK, uN, normalK, normalN);
	  }
	}
	
	// Flux contribution = Kels2D(:,l)'*fstar
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
		    nFN, nStates, nQ2D, 1.0, &Kels2D(0,0,iF/2), nQ2D, 
		    &fStar(0,0,iF,l), nQ2D, 
		    0.0, &faceContributions(0,0,iF,l), nFN);
	
	// Add up face contributions into global residuals array
	for (int iS = 0; iS < nStates; ++iS) {
	  for (int iFN = 0; iFN < nFN; ++iFN) {
	    residuals(mesh.efToN(iFN,iF), iS, iK, l) += faceContributions(iFN, iS, iF, l);
	  }
	}
      }
      
    }
  }
  
}

/** Evaluates the local DG flux function for this PDE at a given value uHere */
inline double Solver::numericalFluxL(double uK, double uN, double normalK, double normalN) const {
  
  auto fK = uK;
  auto fN = uN;
  
  // TODO: In Lax-Friedrichs formulation, this appears to also be the Roe A value?
  //double C = std::abs((fN-fK)/(uN-uK));
  //double result = (fK+fN)/2.0 + (C/2.0)*(uN*normalN + uK*normalK);
  
  double result = (normalK < 0.0 ? fK : fN)*normalK;
  return result;
  
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
  
  // Loop over all elements
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    
    // Initialize flux along faces
    darray fStar{nQ2D, nStates, Mesh::N_FACES}; // TODO: should we move this out of for loop
    darray faceContributions{nFN, nStates, Mesh::N_FACES};
    
    // For every face, compute fstar = fc*(Interp2D*u)
    for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
      
      auto nF = mesh.eToF(iF, iK);
      auto nK = mesh.eToE(iF, iK);
      
      darray normalK{&mesh.normals(0, iF, iK), 3};
      darray normalN{&mesh.normals(0, nF, nK), 3};
      
      // Must compute nStates of these flux integrals per face
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iFQ = 0; iFQ < nQ2D; ++iFQ) {
	  auto uK = uInterp2D(iFQ, iF, iS, iK);
	  auto uN = uInterp2D(iFQ, nF, iS, nK);
	  
	  fStar(iFQ, iS, iF) = numericalFluxC(uK, uN, normalK, normalN);
	}
      }
      
      // Flux contribution = Kels2D(:,l)'*fstar
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
		  nFN, nStates, nQ2D, 1.0, &Kels2D(0,0,iF/2), nQ2D, 
		  &fStar(0,0,iF), nQ2D, 
		  0.0, &faceContributions(0,0,iF), nFN);
      
      // Add up face contributions into global residual array
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iFN = 0; iFN < nFN; ++iFN) {
	  residual(mesh.efToN(iFN,iF), iS, iK) += faceContributions(iFN, iS, iF);
	}
      }
    }
    
  }
  
}

/** Evaluates the convection DG flux function for this PDE using upwinding */
inline double Solver::numericalFluxC(double uK, double uN, const darray& normalK, const darray& normalN) const {
  
  double result = 0.0;
  
  // TODO: fc(u) = a*u right now. This depends on PDE!
  for (int l = 0; l < Mesh::DIM; ++l) {
    auto fK = fluxC(uK, l);
    auto fN = fluxC(uN, l);
    // upwinding assuming a[l] is always positive
    result += (normalK(l) > 0.0 ? fK : fN)*normalK(l);
  }
  
  return result;
  
}

/** Evaluates the actual convection flux function for the PDE */
inline double Solver::fluxC(double uK, int l) const {
  return a[l]*uK;
}

/** Evaluates the volume integral term for convection in the RHS */
void Solver::convectDGVolume(const darray& uInterp3D, darray& residual) const {
  
  int nQ3D = Interp3D.size(0);
  
  // residual += Kels(:, l)*fc_l(uInterp3D)
  darray fc{nQ3D, nStates, mesh.nElements, Mesh::DIM};
  for (int l = 0; l < Mesh::DIM; ++l) {
    
    for (int k = 0; k < mesh.nElements; ++k) {
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iQ = 0; iQ < nQ3D; ++iQ) {
	  fc(iQ,iS,k,l) = fluxC(uInterp3D(iQ,iS,k), l);
	}
      }
    }
    
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		dofs, nStates*mesh.nElements, nQ3D, 1.0, &Kels(0,0,l), nQ3D,
		&fc(0,0,0,l), nQ3D, 1.0, residual.data(), dofs);
    
  }
  
}


/**
   Viscous DG Flux: Computes Fv(u) for use in the Local DG formulation of the 
   2nd-order diffusion term.
   Uses upwinding for the numerical flux. TODO: Use Per's Roe solver.
   Updates residual variable with added flux
*/
void Solver::viscousDGFlux(const darray& uInterp2D, const darray& DuInterp2D, darray& residual) const {
  
  int nFN = mesh.nFNodes;
  int nQ2D = mesh.nFQNodes;
  
  // Loop over all elements
  for (int iK = 0; iK < mesh.nElements; ++iK) {
    
    // Initialize flux along faces
    darray fStar{nQ2D, nStates, Mesh::N_FACES}; // TODO: should we move this out of for loop
    darray faceContributions{nFN, nStates, Mesh::N_FACES};
    
    // For every face, compute flux = integrate over face (fstar*phi_i)
    for (int iF = 0; iF < Mesh::N_FACES; ++iF) {
      
      auto nF = mesh.eToF(iF, iK);
      auto nK = mesh.eToE(iF, iK);
      
      darray normalK{&mesh.normals(0, iF, iK), 3};
      darray normalN{&mesh.normals(0, nF, nK), 3};
      
      // Must compute nStates of these flux integrals per face
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iFQ = 0; iFQ < nQ2D; ++iFQ) {
	  auto uK = uInterp2D(iFQ, iF, iS, iK);
	  auto uN = uInterp2D(iFQ, nF, iS, nK);
	  darray DuK{Mesh::DIM};
	  darray DuN{Mesh::DIM};
	  for (int l = 0; l < Mesh::DIM; ++l) {
	    DuK(l) = DuInterp2D(iFQ, iF, iS, iK, l);
	    DuN(l) = DuInterp2D(iFQ, nF, iS, nK, l);
	  }
	  
	  fStar(iFQ, iS, iF) = numericalFluxV(uK, uN, DuK, DuN, normalK, normalN);
	}
      }
      
      // Flux contribution = Kels2D(:,l)'*fstar
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
		  nFN, nStates, nQ2D, 1.0, &Kels2D(0,0,iF/2), nQ2D, 
		  &fStar(0,0,iF), nQ2D, 
		  0.0, &faceContributions(0,0,iF), nFN);
      
      // Add up face contributions into global residual array
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iFN = 0; iFN < nFN; ++iFN) {
	  residual(mesh.efToN(iFN,iF), iS, iK) += faceContributions(iFN, iS, iF);
	}
      }
    }
    
  }
  
}

/** Evaluates the volume integral term for viscosity in the RHS */
void Solver::viscousDGVolume(const darray& uInterp3D, const darray& DuInterp3D, darray& residual) const {
  
  int nQ3D = Interp3D.size(0);
  
  // residual += Kels(:, l)*fv_l(uInterp3D)
  darray fv{nQ3D, nStates, mesh.nElements, Mesh::DIM};
  for (int l = 0; l < Mesh::DIM; ++l) {
    
    for (int k = 0; k < mesh.nElements; ++k) {
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iQ = 0; iQ < nQ3D; ++iQ) {

	  auto uK = uInterp3D(iQ,iS,k);
	  darray DuK{Mesh::DIM};
	  for (int l2 = 0; l2 < Mesh::DIM; ++l2) {
	    DuK(l2) = DuInterp3D(iQ,iS,k,l2);
	  }
	  fv(iQ,iS,k,l) = fluxV(uK, DuK, l);
	}
      }
    }
    
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		dofs, nStates*mesh.nElements, nQ3D, 1.0, &Kels(0,0,l), nQ3D,
		&fv(0,0,0,l), nQ3D, 1.0, residual.data(), dofs);
    
  }
  
}

/** Evaluates the viscous DG flux function for this PDE using upwinding */
inline double Solver::numericalFluxV(double uK, double uN, const darray& DuK, const darray& DuN, const darray& normalK, const darray& normalN) const {
  
  double result = 0.0;
  
  for (int l = 0; l < Mesh::DIM; ++l) {
    auto fK = fluxV(uK, DuK, l);
    auto fN = fluxV(uN, DuN, l);
    // upwinding assuming a[l] is always positive
    result += (normalK(l) > 0.0 ? fK : fN)*normalK(l);
  }
  
  return result;
  
}

/** Evaluates the actual viscosity flux function for the PDE */
inline double Solver::fluxV(double uK, const darray& DuK, int l) const {
  double eps = 1e-2;
  // TODO: fv(u,Du) = a*u-eps*sum(DuK) right now. This depends on PDE!
  // TODO: change this to convection diffusion
  switch(l) {
  case 0:
    return - eps*a[l]*DuK(l);
    break;
  case 1:
    return - eps*a[l]*DuK(l);
      //(DuK(0) + DuK(1) + DuK(2));
    break;
  case 2:
    return - eps*a[l]*DuK(l);
    break;
  default:
    std::cerr << "How the hell...?" << std::endl;
    return 0.0;
  }
}
