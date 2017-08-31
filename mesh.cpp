#include "mesh.h"

/** Point functionality */
Point::Point() : Point{0.0, 0.0, 0.0} {}

Point::Point(double _x, double _y) : Point{_x, _y, 0.0} {}

Point::Point(double _x, double _y, double _z) : x{_x}, y{_y}, z{_z} { }

Point::Point(const Point& other) {
  x = other.x;
  y = other.y;
  z = other.z;
}

Point& Point::operator=(const Point& p) {
  if (this != &p) {
    x = p.x;
    y = p.y;
    z = p.z;
  }
  return *this;
}

double Point::dist(const Point& other) const {
  return std::sqrt((other.x-x)*(other.x-x) + (other.y-y)*(other.y-y) + (other.z-z)*(other.z-z));
}

std::ostream& operator<<(std::ostream& out, const Point& p) {
  return out << "{" << p.x << ", " << p.y  << ", " << p.z << "}";
}
inline bool operator==(const Point& lhs, const Point& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z);
}
inline bool operator!=(const Point& lhs, const Point& rhs) {
  return !(lhs == rhs);
}

/** End Point functionality */





/** Mesh Functionality */
Mesh::Mesh() : Mesh{10,10} {}

Mesh::Mesh(int nx, int ny) : Mesh{nx, ny, Point{0.0, 0.0}, Point{1.0, 1.0}, MPIUtil{}} {}

Mesh::Mesh(const MPIUtil& _mpi) : Mesh{10, 10, Point{0.0, 0.0}, Point{1.0, 1.0}, _mpi} {}

/** Main constructor */
Mesh::Mesh(int nx, int ny, const Point& _botLeft, const Point& _topRight, const MPIUtil& _mpi) :
  mpi{_mpi},
  nElements{},
  nIElements{},
  nBElements{},
  nGElements{},
  mpiNBElems{},
  nVertices{},
  botLeft{_botLeft},
  topRight{_topRight},
  dxMin{},
  dxMax{},
  order{},
  globalCoords{},
  globalQuads{},
  vertices{},
  eToV{},
  fToV{},
  beToE{},
  ieToE{},
  mpibeToE{},
  eToE{},
  eToF{},
  normals{},
  bilinearMapping{},
  efToN{},
  nfnToFN{},
  efToQ{},
  nfqToFQ{}
{
  defaultSquare(nx, ny);
}

Mesh::Mesh(const std::string& filename, const MPIUtil& _mpi) :
  mpi{_mpi},
  nElements{},
  nIElements{},
  nBElements{},
  nGElements{},
  mpiNBElems{},
  nVertices{},
  botLeft{},
  topRight{},
  dxMin{},
  dxMax{},
  order{},
  globalCoords{},
  globalQuads{},
  vertices{},
  eToV{},
  fToV{},
  beToE{},
  ieToE{},
  mpibeToE{},
  eToE{},
  eToF{},
  normals{},
  bilinearMapping{},
  efToN{},
  nfnToFN{},
  efToQ{},
  nfqToFQ{}
{
  
  // Read mesh from file
  iarray periodicity;
  bool success = readMesh(filename, DIM, N_VERTICES,
  	   nVertices, nElements, vertices, eToV, periodicity);
  if (!success) {
    mpi.exit(-1);
  }
  
  /* TODO: debugging
  srand(time(NULL));
  
  for (int i = 0; i < nVertices; ++i) {
    if (i <= 11 || i%11 == 0 || i%11 == 10 || i>100) {
      continue;
    }
    for (int l = 0; l < DIM; ++l) {
      vertices(l,i) += ((.04*(float)rand()/(float)RAND_MAX)-.02);
    }
  }
  */
  
  initFToV();
  initBilinearMappings();
  initNormals();
  
  // Must be done after initializing normals and bilinear maps
  resolvePeriodicities(periodicity);
  
  
  // TODO: make this MPI compatible
  nIElements = nElements;
  nBElements = 0;
  nGElements = 0;
  mpiNBElems.realloc(MPIUtil::N_FACES);
  for (int iF = 0; iF < MPIUtil::N_FACES; ++iF) {
    mpiNBElems(iF) = 0;
  }
  beToE.realloc(nBElements);
  ieToE.realloc(nIElements);
  for (int iK = 0; iK < nElements; ++iK) {
    ieToE(iK) = iK;
  }
  mpibeToE.realloc(max(mpiNBElems), MPIUtil::N_FACES);
  
  // Initialize eToE and eToF
  initConnectivity();
  
}

void Mesh::defaultSquare(int nx, int ny) {
  
  // Initialize my position in MPI topology
  int globalNs[MPIUtil::DIM] = {nx, ny};
  int localNs[MPIUtil::DIM];
  int localSs[MPIUtil::DIM];
  int localEs[MPIUtil::DIM];
  for (int l = 0; l < MPIUtil::DIM; ++l) {
    
    localNs[l] = static_cast<int>(std::floor(globalNs[l] / mpi.nps[l]));
    if(mpi.coords[l] < globalNs[l] % mpi.nps[l]) {
      localNs[l]++;
      localSs[l] = mpi.coords[l] * localNs[l];
      localEs[l] = (mpi.coords[l] + 1)*localNs[l];
    } else {
      localSs[l] = (globalNs[l] % mpi.nps[l])*(localNs[l]+1) 
	+ (mpi.coords[l]     - (globalNs[l]%mpi.nps[l]))*localNs[l];
      localEs[l] = (globalNs[l] % mpi.nps[l])*(localNs[l]+1) 
	+ (mpi.coords[l] + 1 - (globalNs[l]%mpi.nps[l]))*localNs[l];
    }
    
  }
  
  for (int l = 0; l < MPIUtil::DIM; ++l) {
    if (localNs[l] < 3) {
      std::cerr << "ERROR: on rank " << mpi.rank << ": " << l << " MPI dimension is not enough to have interior elements!" << std::endl;
      mpi.exit(-1);
    }
  }
  
  // Distinguishing between boundary and interior elements
  nElements  = localNs[0]*localNs[1];
  nIElements = (localNs[0]-2)*(localNs[1]-2);
  nBElements = nElements - nIElements;
  nGElements = 0;
  mpiNBElems.realloc(MPIUtil::N_FACES);
  for (int iF = 0; iF < MPIUtil::N_FACES; ++iF) {
    //mpiNBElems(iF) = 0;
    //* necessary for periodic BCs
    mpiNBElems(iF) = 1;
    for (int l = 0; l < MPIUtil::DIM; ++l) {
      if (l != iF/2) {
	mpiNBElems(iF) *= localNs[l];
      }
    }
    nGElements += mpiNBElems(iF);
    //*/
  }
  
  nVertices  = (localNs[0]+1)*(localNs[1]+1);
  
  // Initialize vertices
  vertices.realloc(DIM, localNs[0]+1, localNs[1]+1);
  for (int iy = localSs[1]; iy <= localEs[1]; ++iy) {
    double currY = iy*(topRight.y-botLeft.y)/ny + botLeft.y;
    int iy0 = iy - localSs[1];
    
    for (int ix = localSs[0]; ix <= localEs[0]; ++ix) {
      double currX = ix*(topRight.x-botLeft.x)/nx + botLeft.x;
      int ix0 = ix - localSs[0];
      
      vertices(0, ix0,iy0) = currX;
      vertices(1, ix0,iy0) = currY;
    }
  }
  vertices.resize(DIM, nVertices);
  
  // Initialize elements-to-vertices array
  eToV.realloc(N_VERTICES, nElements);
  for (int iy = 0; iy < localNs[1]; ++iy) {
    int yOff1 = (iy  )*(localNs[0]+1);
    int yOff2 = (iy+1)*(localNs[0]+1);
    for (int ix = 0; ix < localNs[0]; ++ix) {
      int eIndex = ix + iy*localNs[0];
      int xOff1 = ix;
      int xOff2 = ix+1;
      
      eToV(0, eIndex) = xOff1+yOff1;
      eToV(1, eIndex) = xOff2+yOff1;
      eToV(2, eIndex) = xOff2+yOff2;
      eToV(3, eIndex) = xOff1+yOff2;
    }
  }
  
  // Initialize boundary element maps
  beToE.realloc(nBElements);
  ieToE.realloc(nIElements);
  int bIndex = 0;
  int iIndex = 0;
  for (int iy = 0; iy < localNs[1]; ++iy) {
    for (int ix = 0; ix < localNs[0]; ++ix) {
      int eIndex = ix + iy*localNs[0];
      if (ix == 0 || ix == localNs[0]-1 || iy == 0 || iy == localNs[1]-1) {
	beToE(bIndex) = eIndex;
	bIndex++;
      }
      else {
	ieToE(iIndex) = eIndex;
	iIndex++;
      }
    }
  }
  
  // Initialize element-to-face arrays and MPI boundary element map
  eToE.realloc(N_FACES, nElements);
  eToF.realloc(N_FACES, nElements);
  mpibeToE.realloc(max(mpiNBElems), MPIUtil::N_FACES);
  
  iarray faceOffsets{MPIUtil::N_FACES};
  int offset = 0;
  for (int l = 0; l < MPIUtil::N_FACES; ++l) {
    faceOffsets(l) = nElements+offset;
    offset += mpiNBElems(l);
  }
  
  // Periodic boundary conditions enforced automatically through MPI cartesian topology
  for (int iy = 0; iy < localNs[1]; ++iy) {
    
    bool face2 = false;
    bool face3 = false;
    int iyM = iy-1;
    if (iyM < 0) { // face 2
      face2 = true;
    }
    iyM *= localNs[0];
    
    int iyP = iy+1;
    if (iyP >= localNs[1]) { // face 3
      face3 = true;
    }
    iyP *= localNs[0];
    int iy0 = iy*localNs[0];
    
    for (int ix = 0; ix < localNs[0]; ++ix) {
      bool face0 = false;
      bool face1 = false;
      int ixM = ix-1;
      if (ixM < 0) { // face 0
	face0 = true;
      }
      int ixP = ix+1;
      if (ixP >= localNs[0]) { // face 1
	face1 = true;
      }
      int ix0 = ix;
      
      int eIndex = ix0 + iy0;
      
      // Neighbor elements in -x,+x,-y,+y directions stored in faces
      if (face0) {
	// Apply absorbing boundary condition at -x boundary
	//eToE(0, eIndex) = static_cast<int>(Boundary::absorbing);
	
	int ghostNum = iy;
	eToE(0, eIndex) = faceOffsets(0)+ghostNum;
	mpibeToE(ghostNum, 0) = eIndex;
      }
      else {
	eToE(0, eIndex) = ixM+iy0;
      }
      
      if (face1) {
	// Apply absorbing boundary condition at +x boundary
	//eToE(1, eIndex) = static_cast<int>(Boundary::absorbing);
	
	int ghostNum = iy;
	eToE(1, eIndex) = faceOffsets(1)+ghostNum;
	mpibeToE(ghostNum, 1) = eIndex;
      }
      else {
	eToE(1, eIndex) = ixP+iy0;
      }
      
      if (face2) {
	// Apply absorbing boundary condition at -y boundary
	//eToE(2, eIndex) = static_cast<int>(Boundary::absorbing);
	
	int ghostNum = ix;
	eToE(2, eIndex) = faceOffsets(2)+ghostNum;
	mpibeToE(ghostNum, 2) = eIndex;
      }
      else {
	eToE(2, eIndex) = ix0+iyM;
      }
      
      if (face3) {
	// Apply free-surface boundary condition at +y boundary
	//eToE(3, eIndex) = static_cast<int>(Boundary::free);
	
	int ghostNum = ix;
	eToE(3, eIndex) = faceOffsets(3)+ghostNum;
	mpibeToE(ghostNum, 3) = eIndex;
      }
      else {
	eToE(3, eIndex) = ix0+iyP;
      }
      
    }
  }
  
  for (int iy = 0; iy < localNs[1]; ++iy) {
    for (int ix = 0; ix < localNs[0]; ++ix) {
      int eIndex = ix + iy*localNs[0];
      
      // Face ID of this element's -x face will be neighbor's +x face (same for all dimensions)
      eToF(0, eIndex) = 1;
      eToF(1, eIndex) = 0;
      eToF(2, eIndex) = 3;
      eToF(3, eIndex) = 2;
      
    }
  }
  
  // Modify fToV to handle default square face connectivity
  fToV.realloc(N_FVERTICES, N_FACES);
  fToV(0,0) = 3; // -x
  fToV(1,0) = 0;
  fToV(0,1) = 1; // +x
  fToV(1,1) = 2;
  fToV(0,2) = 0; // -y
  fToV(1,2) = 1;
  fToV(0,3) = 2; // +y
  fToV(1,3) = 3;
  
  initBilinearMappings();
  
  initNormals();
  
}

/** Copy constructor */
Mesh::Mesh(const Mesh& other) :
  mpi{other.mpi},
  nElements{other.nElements},
  nIElements{other.nIElements},
  nBElements{other.nBElements},
  nGElements{other.nGElements},
  mpiNBElems{other.mpiNBElems},
  nVertices{other.nVertices},
  botLeft{other.botLeft},
  topRight{other.topRight},
  dxMin{other.dxMin},
  dxMax{other.dxMax},
  order{other.order},
  globalCoords{other.globalCoords},
  globalQuads{other.globalQuads},
  vertices{other.vertices},
  eToV{other.eToV},
  fToV{other.fToV},
  beToE{other.beToE},
  ieToE{other.ieToE},
  mpibeToE{other.mpibeToE},
  eToE{other.eToE},
  eToF{other.eToF},
  normals{other.normals},
  bilinearMapping{other.bilinearMapping},
  efToN{other.efToN},
  nfnToFN{other.nfnToFN},
  efToQ{other.efToQ},
  nfqToFQ{other.nfqToFQ}
{ }

/** Label faces of rectangle according to vertices */
void Mesh::initFToV() {
  
  // Assumes input points are CCW
  fToV.realloc(N_FVERTICES,N_FACES);
  for (int iF = 0; iF < N_FACES; ++iF) {
    fToV(0,iF) = iF;
    fToV(1,iF) = (iF+1 == N_FACES ? 0 : iF+1);
  }
  
}

/** Initialize bilinear mappings from element coordinates */
void Mesh::initBilinearMappings() {
  
  // Sanity check: mapping works with this element type
  int mapOrder = 1;
  int mapDofs = static_cast<int>(std::pow((mapOrder+1), DIM));
  if (mapDofs != N_VERTICES) { // better be exactly equal to number of vertices
    std::cerr << "FATAL ERROR: trying to initialize with a map with too high an order!" << std::endl;
    mpi.exit(-1);
  }
  
  // Gmsh =   3---2   Mapping =  2---3
  // CCW      |   |   wants      |   |
  // corners  0---1              0---1
  darray c2m{N_VERTICES}; // corner to mapping
  c2m(0) = 0; c2m(1) = 1;
  c2m(2) = 3; c2m(3) = 2;
  
  bilinearMapping.realloc(N_VERTICES, DIM, nElements);
  for (int iK = 0; iK < nElements; ++iK) {
    for (int l = 0; l < DIM; ++l) {
      for (int iV = 0; iV < N_VERTICES; ++iV) {
	bilinearMapping(c2m(iV),l,iK) = vertices(l,eToV(iV,iK));
      }
    }
  }
  
}

/** Initialize normals from element coordinates. Assumes input 2D mesh is ordered CCW */
void Mesh::initNormals() {
  
  // Simultaneously computes dxMin/dxMax
  dxMin = std::numeric_limits<double>::max();
  dxMax = 0.0;
  
  normals.realloc(DIM, N_FACES, nElements);
  for (int iK = 0; iK < nElements; ++iK) {
    for (int iF = 0; iF < N_FACES; ++iF) {
      int a = fToV(0,iF);
      int b = fToV(1,iF);
      
      double ABx = vertices(0,eToV(b,iK)) - vertices(0,eToV(a,iK));
      double ABy = vertices(1,eToV(b,iK)) - vertices(1,eToV(a,iK));
      double length = std::sqrt(ABx*ABx + ABy*ABy);
      normals(0, iF, iK) = ABy/length;
      normals(1, iF, iK) = -ABx/length;
      
      if (length > dxMax)
	dxMax = length;
      if (length < dxMin)
	dxMin = length;
      
    }
  }
  
  // TODO: with isoparametric mapping, normals should be at quadrature points
  // look in 3dg/src/dgmodel.cpp/precomp_edgeJnn for how Per sets up 3dg's nns = normals(Mesh::DIM, nQF, N_FACES, nElements)
  // called by 3dg/src/dgassemble.cpp/Tassemble::el_setup
  
}

/** Resolve periodicities in eToV */
void Mesh::resolvePeriodicities(const iarray& periodicity) {
  
  bool resolved = false;
  while (!resolved) {
    
    resolved = true;
    
    for (int i = 0; i < nVertices; ++i) {
      if (periodicity(i) != i) {
	// Found a vertex renumbering based off of periodic BC
	for (int iK = 0; iK < nElements; ++iK) {
	  for (int iV = 0; iV < N_VERTICES; ++iV) {
	    if (eToV(iV,iK) == i) {
	      // Found a vertex in eToV that should be changed
	      eToV(iV,iK) = periodicity(i);
	      resolved = false;
	    }
	  }
	}
      }
    }
    
  }
  
}

/** Initialize connectivity in eToE and eToF */
void Mesh::initConnectivity() {
  
  // sequential O(n^2) method for determining eToE/eToF
  eToE.realloc(N_FACES, nElements);
  eToF.realloc(N_FACES, nElements);
  barray facesSeen{N_FACES, nElements};
  for (int iK = 0; iK < nElements; ++iK) {
    for (int iF = 0; iF < N_FACES; ++iF) {
      facesSeen(iF,iK) = false;
    }
  }
  
  // Loop over all of the element/faces
  for (int iK = 0; iK < nElements; ++iK) {
    for (int iF = 0; iF < N_FACES; ++iF) {
      if (facesSeen(iF,iK)) {
	continue;
      } else {
	facesSeen(iF,iK) = true;
      }
      // Visiting this current element/face
      std::set<int> iFace;
      for (int iFV = 0; iFV < N_FVERTICES; ++iFV) {
	iFace.insert(eToV(fToV(iFV,iF),iK));
      }
      
      bool boundary = true;
      
      // Compare with all possible neighbor elements/faces
      for (int nK = 0; nK < nElements; ++nK) {
	for (int nF = 0; nF < N_FACES; ++nF) {
	  if (facesSeen(nF,nK))
	    continue;
	  
	  std::set<int> nFace;
	  for (int nFV = 0; nFV < N_FVERTICES; ++nFV) {
	    nFace.insert(eToV(fToV(nFV,nF),nK));
	  }
	  
	  if (iFace == nFace) {
	    // Link both elements
	    eToE(iF,iK) = nK;
	    eToE(nF,nK) = iK;
	    
	    eToF(iF,iK) = nF;
	    eToF(nF,nK) = iF;
	    
	    facesSeen(nF,nK) = true;
	    
	    boundary = false;
	    break;
	  }
	  
	}
	if (!boundary)
	  break;
      }
      
      if (boundary) {
	
	std::cout << "found a boundary at " << iK << ", face " << iF << std::endl;
	
	double y1 = vertices(1, eToV(fToV(0,iF), iK));
	double y2 = vertices(1, eToV(fToV(1,iF), iK));
	if (y1 == 0.0 && y2 == 0.0) {
	  // north face has free surface condition
	  eToE(iF,iK) = static_cast<int>(Boundary::free);
	}
	else {
	  // all other sides have absorbing BC
	  eToE(iF,iK) = static_cast<int>(Boundary::absorbing);
	}
	eToF(iF,iK) = -1; // not to be accessed
	
      }
      
    }
  }
  
}

/** Initialize global nodes from bilinear mapping of reference nodes */
void Mesh::setupNodes(const darray& InterpTk, int _order) {
  
  order = _order;
  int nNodes = InterpTk.size(0);
  
  darray xl{nNodes};
  // Scales and translates nodes into each element by applying bilinear mapping
  globalCoords.realloc(DIM, nNodes, nElements);
  for (int iK = 0; iK < nElements; ++iK) {
    for (int l = 0; l < DIM; ++l) {
      // coord_l = InterpTk*bilinear(:,l,iK)
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		  nNodes,1,N_VERTICES, 1.0, InterpTk.data(), nNodes,
		  &bilinearMapping(0,l,iK), N_VERTICES, 0.0, xl.data(), nNodes);
      for (int iN = 0; iN < nNodes; ++iN) {
	globalCoords(l,iN,iK) = xl(iN);
      }
    }
  }
  
  // Nodes per face
  int nFNodes = initFaceMap(efToN, nfnToFN, order+1); 
  
}

/** Initialize global quadrature points from bilinear mappings of reference quadrature points */
void Mesh::setupQuads(const darray& InterpTkQ, int nQV) {
  
  globalQuads.realloc(DIM, nQV, nElements);
  darray xl{nQV};
  for (int iK = 0; iK < nElements; ++iK) {
    for (int l = 0; l < DIM; ++l) {
      // coord_l = InterpTkQ*bilinear(:,l,iK)
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		  nQV,1,N_VERTICES, 1.0, InterpTkQ.data(), nQV,
		  &bilinearMapping(0,l,iK), N_VERTICES, 0.0, xl.data(), nQV);
      for (int iQ = 0; iQ < nQV; ++iQ) {
	globalQuads(l,iQ,iK) = xl(iQ);
      }
    }
  }
  
  // Quadrature points per face
  int nQ = (int)std::ceil(order+1/2.0);
  int nFQNodes = initFaceMap(efToQ, nfqToFQ, nQ);
  mpi.initDatatype(nFQNodes);
  
}

/**
   Initialize face maps for a given number of nodes.
   Assumes that each face is a side with exactly size1D nodes.
   
   For every element k, face i, face node iFN:
   My node @: soln(efMap(iFN, i), :, k)
   Neighbor's node @: soln(efMap(nfMap(iFN), eToF(i,k)), :, eToE(i,k))
*/
int Mesh::initFaceMap(iarray& efMap, iarray& nfMap, int size1D) {
  
  efMap.realloc(size1D, N_FACES);
  nfMap.realloc(size1D);
  
  int xOff;
  int yOff;
  
  // efMap has explicit dependence on fToV
  // Note: only handles CCW vertices/faces
  // 3---2
  // |   |
  // 0---1
  // Care is given to ensure nodes on each face are CCW
  for (int iF = 0; iF < N_FACES; ++iF) {
    if (fToV(0,iF) == 3 && fToV(1,iF) == 0) {
      // -x direction face
      xOff = 0;
      for (int iy = 0; iy < size1D; ++iy) {
	yOff = (size1D-iy-1)*size1D;
	efMap(iy, iF) = xOff+yOff;
      }
    }
    else if (fToV(0,iF) == 0 && fToV(1,iF) == 1) {
      // -y direction face
      yOff = 0;
      for (int ix = 0; ix < size1D; ++ix) {
	xOff = ix;
	efMap(ix, iF) = xOff+yOff;
      }
    }
    else if (fToV(0,iF) == 1 && fToV(1,iF) == 2) {
      // +x direction face
      xOff = size1D-1;
      for (int iy = 0; iy < size1D; ++iy) {
	yOff = iy*size1D;
	efMap(iy, iF) = xOff+yOff;
      }
    }
    else if (fToV(0,iF) == 2 && fToV(1,iF) == 3) {
      // +y direction face
      yOff = (size1D-1)*size1D;
      for (int ix = 0; ix < size1D; ++ix) {
	xOff = (size1D-ix-1);
	efMap(ix, iF) = xOff+yOff;
      }
    }
    else {
      std::cerr << "ERROR: Vertices for fToV are not CCW!" << std::endl;
      mpi.exit(-1);
    }
  }
  
  // nfMap is CW-oriented version of efMap
  for (int i = 0; i < size1D; ++i) {
    nfMap(i) = size1D-i-1;
  }
  
  // Re-organize face nodes so they are accessible by one index
  int tempFNodes = size1D;
  efMap.resize(tempFNodes, N_FACES);
  return tempFNodes;
  
}


/**
   Initializes absolute value of determinant of Jacobian of mappings
   calculated at the quadrature points xQV
   within the volume stored in Jk and along the faces stored in JkF
*/
void Mesh::setupJacobians(int nQV, const darray& xQV, darray& Jk, darray& JkInv,
			  int nQF, const darray& xQF, darray& JkF) const {
  
  int mapOrder = 1;
  
  // Compute gradient of mapping reference bases on the quadrature points
  darray chebyTk;
  chebyshev2D(mapOrder, chebyTk);
  darray dPhiTkQ;
  dPhi2D(chebyTk, xQV, dPhiTkQ);
  
  // Explicitly form the face quadrature points
  darray chebyTkF;
  chebyshev1D(mapOrder, chebyTkF);
  darray xQF2D{DIM, nQF, N_FACES};
  
  // Below mappings based off of fToV explicitly
  for (int iF = 0; iF < N_FACES; ++iF) {
    if (fToV(0,iF) == 3 && fToV(1,iF) == 0) {
      // along xi_0 = -1
      for (int iQ = 0; iQ < nQF; ++iQ) {
	xQF2D(0,iQ,iF) = chebyTkF(0,0);
	xQF2D(1,iQ,iF) = xQF(0,iQ);
      }
    }
    else if (fToV(0,iF) == 0 && fToV(1,iF) == 1) {
      // along xi_1 = -1
      for (int iQ = 0; iQ < nQF; ++iQ) {
	xQF2D(0,iQ,iF) = xQF(0,iQ);
	xQF2D(1,iQ,iF) = chebyTkF(0,0);
      }
    }
    else if (fToV(0,iF) == 1 && fToV(1,iF) == 2) {
      // along xi_0 = 1
      for (int iQ = 0; iQ < nQF; ++iQ) {
	xQF2D(0,iQ,iF) = chebyTkF(0,1);
	xQF2D(1,iQ,iF) = xQF(0,iQ);
      }
    }
    else if (fToV(0,iF) == 2 && fToV(1,iF) == 3) {
      // along xi_1 = 1
      for (int iQ = 0; iQ < nQF; ++iQ) {
	xQF2D(0,iQ,iF) = xQF(0,iQ);
	xQF2D(1,iQ,iF) = chebyTkF(0,1);
      }
    }
    else {
      std::cerr << "ERROR: Vertices for fToV are not CCW!" << std::endl;
      mpi.exit(-1);
    }
  }
  
  
  // Compute gradient of mapping reference bases on the face quadrature points
  darray dPhiTkQF;
  dPhi2D(chebyTk, xQF2D, dPhiTkQF);
  
  // Initialize Jacobians
  darray JacobianK{nQV, DIM, DIM};
  darray JacobianKF{nQF,N_FACES, DIM, DIM};
  Jk.realloc(nQV,nElements);
  JkF.realloc(nQF,N_FACES,nElements);
  JkInv.realloc(DIM,DIM,nQV,nElements);
  
  for (int iK = 0; iK < nElements; ++iK) {
    
    // Compute Jacobian = dPhiTkQ(:,:,l_j)*bilinearMapping(:,l_i,iK)
    for (int lj = 0; lj < DIM; ++lj) {
      for (int li = 0; li < DIM; ++li) {
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
		    nQV,1,N_VERTICES, 1.0, &dPhiTkQ(0,0,lj), nQV, 
		    &bilinearMapping(0,li,iK), Mesh::N_VERTICES, 
		    0.0, &JacobianK(0,li,lj), nQV);
      }
    }
    
    // Compute Jk = |det(Jacobian)| at each quadrature point
    // and JkInv  = Jacobian^{-1} at each quadarture point
    for (int iQ = 0; iQ < nQV; ++iQ) {
      Jk(iQ,iK) = JacobianK(iQ,0,0)*JacobianK(iQ,1,1) 
	        - JacobianK(iQ,1,0)*JacobianK(iQ,0,1);
#ifdef DEBUG
      double eps = 1e-16;
      if (std::abs(Jk(iQ,iK)) < eps) {
	std::cerr << "ERROR: Jacobian = 0 for mapping of mesh element " << iK << "!" << std::endl;
	mpi.exit(-1);
      }
#endif
      
      JkInv(0,0,iQ,iK) =  JacobianK(iQ,1,1)/Jk(iQ,iK);
      JkInv(1,0,iQ,iK) = -JacobianK(iQ,1,0)/Jk(iQ,iK);
      JkInv(0,1,iQ,iK) = -JacobianK(iQ,0,1)/Jk(iQ,iK);
      JkInv(1,1,iQ,iK) =  JacobianK(iQ,0,0)/Jk(iQ,iK);
      
      Jk(iQ,iK) = std::abs(Jk(iQ,iK));
    }
    
    
    // Compute Jacobian = dPhiTkQF(:,:,l_j)*bilinearMapping(:,l_i,iK)
    for (int lj = 0; lj < DIM; ++lj) {
      for (int li = 0; li < DIM; ++li) {
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
		    nQF*N_FACES, 1, N_VERTICES, 1.0, 
		    &dPhiTkQF(0,0,lj), nQF*N_FACES, 
		    &bilinearMapping(0,li,iK), Mesh::N_VERTICES, 
		    0.0, &JacobianKF(0,0,li,lj), nQF*N_FACES);
      }
    }
    
    // TODO: think about this carefully, should JkF = ||tangent||/2?
    // 1D integral => Only have a |gamma'(t)| term along correct xi_{l_i}
    // Compute JkF = |gamma'(t)| along at each face quadrature point
    int lj;
    for (int iF = 0; iF < N_FACES; ++iF) {
      
      if (fToV(0,iF) == 3 && fToV(1,iF) == 0) {
	// along xi_0 = -1
	lj = 1;
      }
      else if (fToV(0,iF) == 0 && fToV(1,iF) == 1) {
	// along xi_1 = -1
	lj = 0;
      }
      else if (fToV(0,iF) == 1 && fToV(1,iF) == 2) {
	// along xi_0 = 1
	lj = 1;
      }
      else if (fToV(0,iF) == 2 && fToV(1,iF) == 3) {
	// along xi_1 = 1
	lj = 0;
      }
      else {
	std::cerr << "ERROR: Vertices for fToV are not CCW!" << std::endl;
	mpi.exit(-1);
      }
      
      for (int iQ = 0; iQ < nQF; ++iQ) {
	JkF(iQ,iF,iK) = std::sqrt(JacobianKF(iQ,iF,0,lj)*JacobianKF(iQ,iF,0,lj) 
				+ JacobianKF(iQ,iF,1,lj)*JacobianKF(iQ,iF,1,lj));
      }
    }
    
  }
  
}

/** Writes mesh to example Gmsh 3.0 .msh file */
void Mesh::outputMesh(const std::string& filename, int nx, int ny) const {
  
  std::ofstream mshFile(filename, std::ios::out);
  if (mshFile.fail()) {
    std::cerr << "ERROR: could not open output mesh " << filename << std::endl;
  }
  
  mshFile << "$MeshFormat\n";
  mshFile << "2.2 0 8\n";
  mshFile << "$EndMeshFormat\n";
  
  // vertices
  mshFile << "$Nodes\n";
  mshFile << nVertices << "\n";
  for (int i = 0; i < nVertices; ++i) {
    mshFile << (i+1) << " " << vertices(0,i) << " " << vertices(1,i) << " " << "0\n";
  }
  mshFile << "$EndNodes\n";
  
  // elements (eToV)
  mshFile << "$Elements\n";
  mshFile << nElements << "\n";
  for (int iK = 0; iK < nElements; ++iK) {
    mshFile << (iK+1) << " " << "3 2 1 1 ";
    
    // Output element corners CCW according to fToV
    int iFPrev = 0;
    mshFile << (eToV(fToV(0,iFPrev),iK)+1) << " "
	    << (eToV(fToV(1,iFPrev),iK)+1) << " ";
    int facesVisited = 1;
    while(facesVisited < N_FACES-1) {
    
      for (int iF = 0; iF < N_FACES; ++iF) {
	if (eToV(fToV(0,iF),iK) == eToV(fToV(1,iFPrev),iK)) {
	  mshFile << (eToV(fToV(1,iF),iK)+1) << " ";
	  
	  // move to next face
	  iFPrev = iF;
	  facesVisited++;
	  break;
	}
      }
      
    }
    mshFile << "\n";
    
  }
  mshFile << "$EndElements\n";
  
  if (nx > 0 && ny > 0) {
    // periodicity of the default square
    iarray periodicity{2,nx+ny+2};
    int iy0 = 0;
    int iy1 = ny;
    for (int ix = 0; ix <= nx; ++ix) {
      periodicity(0,ix) = ix+iy0*(nx+1);
      periodicity(1,ix) = ix+iy1*(nx+1);
    }
    int ix0 = 0;
    int ix1 = nx;
    for (int iy = 0; iy <= ny; ++iy) {
      periodicity(0,nx+1+iy) = ix0+iy*(nx+1);
      periodicity(1,nx+1+iy) = ix1+iy*(nx+1);
    }
    
    mshFile << "$Periodic\n";
    mshFile << "1\n";
    mshFile << "0 1 2\n";
    mshFile << periodicity.size(1) << "\n";
    for (int i = 0; i < periodicity.size(1); ++i) {
      mshFile << (periodicity(0,i)+1) << " " << (periodicity(1,i)+1) << "\n";
    }
    mshFile << "$EndPeriodic\n";
  }
  
}

/** Allows C++ command-line stream output */
std::ostream& operator<<(std::ostream& out, const Mesh& mesh) {
  out << mesh.nVertices << " vertices connected with " << mesh.nElements << " elements." << std::endl;
  return out;
}

