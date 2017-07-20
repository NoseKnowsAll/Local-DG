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
  return out << "{" << p.x << ", " << p.y  << "}";
}
inline bool operator==(const Point& lhs, const Point& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y);
}
inline bool operator!=(const Point& lhs, const Point& rhs) {
  return !(lhs == rhs);
}

/** End Point functionality */





/** Mesh Functionality */
Mesh::Mesh() : Mesh{10,10,10} {}

Mesh::Mesh(int nx, int ny, int nz) : Mesh{nx, ny, nz, Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 1.0}, MPIUtil{}} {}

Mesh::Mesh(const MPIUtil& _mpi) : Mesh{10, 10, 10, Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 1.0}, _mpi} {}

/** Main constructor */
Mesh::Mesh(int nx, int ny, int nz, const Point& _botLeft, const Point& _topRight, const MPIUtil& _mpi) :
  mpi{_mpi},
  nElements{},
  nIElements{},
  nBElements{},
  nGElements{},
  mpiNBElems{},
  nVertices{},
  botLeft{_botLeft},
  topRight{_topRight},
  minDX{},
  minDY{},
  minDZ{},
  order{},
  nNodes{},
  nFNodes{},
  nFQNodes{},
  globalCoords{},
  globalQuads{},
  vertices{},
  eToV{},
  beToE{},
  ieToE{},
  mpibeToE{},
  eToE{},
  eToF{},
  normals{},
  tempMapping{},
  efToN{},
  efToQ{}
{

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
      exit(-1);
    }
  }
  
  // Distinguishing between boundary and interior elements
  nElements  = localNs[0]*localNs[1];
  nIElements = (localNs[0]-2)*(localNs[1]-2);
  nBElements = nElements - nIElements;
  nGElements = 0;
  mpiNBElems.realloc(MPIUtil::N_FACES);
  for (int iF = 0; iF < MPIUtil::N_FACES; ++iF) {
    mpiNBElems(iF) = 1;
    for (int l = 0; l < MPIUtil::DIM; ++l) {
      if (l != iF/2) {
	mpiNBElems(iF) *= localNs[l];
      }
    }
    nGElements += mpiNBElems(iF);
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
  
  minDX = (topRight.x - botLeft.x)/nx;
  minDY = (topRight.y - botLeft.y)/ny;
  
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
      eToV(2, eIndex) = xOff1+yOff2;
      eToV(3, eIndex) = xOff2+yOff2;
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
  
  // */
  // TODO: work on reading mesh
  //readMesh("test.msh", DIM, N_VERTICES,
  //	   nVertices, nElements, vertices, eToV);
  //vertices.realloc(DIM, nVertices);
  //eToV.realloc(N_VERTICES, nElements);
  //beToE.realloc(nBElements);
  //ieToE.realloc(nIElements);
  
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
	int ghostNum = iy;
	eToE(0, eIndex) = faceOffsets(0)+ghostNum;
	mpibeToE(ghostNum, 0) = eIndex;
      }
      else {
	eToE(0, eIndex) = ixM+iy0;
      }
      
      if (face1) {
	int ghostNum = iy;
	eToE(1, eIndex) = faceOffsets(1)+ghostNum;
	mpibeToE(ghostNum, 1) = eIndex;
      }
      else {
	eToE(1, eIndex) = ixP+iy0;
      }
      
      if (face2) {
	int ghostNum = ix;
	eToE(2, eIndex) = faceOffsets(2)+ghostNum;
	mpibeToE(ghostNum, 2) = eIndex;
      }
      else {
	eToE(2, eIndex) = ix0+iyM;
      }
      
      if (face3) {
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
  
  // Initialize normals
  normals.realloc(DIM, N_FACES, nElements);
  for (int iy = 0; iy < localNs[1]; ++iy) {
    for (int ix = 0; ix < localNs[0]; ++ix) {
      int eIndex = ix + iy*localNs[0];
      
      // Note that normals is already filled with 0s
      normals(0, 0, eIndex) = -1.0;
      normals(0, 1, eIndex) = 1.0;
      normals(1, 2, eIndex) = -1.0;
      normals(1, 3, eIndex) = 1.0;
      
    }
  }
  
  // Temporary bad mapping, 0 == scaling factor, 1 == translation
  tempMapping.realloc(2, DIM, nElements);
  for (int k = 0; k < nElements; ++k) {
    for (int l = 0; l < DIM; ++l) {
      double x0 = vertices(l, eToV(0,k));
      double x1 = vertices(l, eToV(N_VERTICES-1,k));
      double dx = x1 - x0;
      
      tempMapping(0, l, k) = dx/2.0;
      tempMapping(1, l, k) = x0+dx/2.0;
    }
  }
  
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
  minDX{other.minDX},
  minDY{other.minDY},
  minDZ{other.minDZ},
  order{other.order},
  nNodes{other.nNodes},
  nFNodes{other.nFNodes},
  nFQNodes{other.nFQNodes},
  globalCoords{other.globalCoords},
  globalQuads{other.globalQuads},
  vertices{other.vertices},
  eToV{other.eToV},
  beToE{other.beToE},
  ieToE{other.ieToE},
  mpibeToE{other.mpibeToE},
  eToE{other.eToE},
  eToF{other.eToF},
  normals{other.normals},
  tempMapping{other.tempMapping},
  efToN{other.efToN},
  efToQ{other.efToQ}
{ }

/** Initialize global nodes from solver's Chebyshev nodes */
void Mesh::setupNodes(const darray& chebyNodes, int _order) {
  
  order = _order;
  nNodes = chebyNodes.size(1)*chebyNodes.size(2);
  darray refNodes{chebyNodes, DIM, nNodes}; 
  
  // Scales and translates Chebyshev nodes into each element
  globalCoords.realloc(DIM, nNodes, nElements);
  for (int k = 0; k < nElements; ++k) {
    for (int iN = 0; iN < nNodes; ++iN) {
      for (int l = 0; l < DIM; ++l) {
	globalCoords(l,iN,k) = tempMapping(0,l,k)*refNodes(l,iN)+tempMapping(1,l,k);
      }
    }
  }
  
  // nodal points per face
  nFNodes = initFaceMap(efToN, order+1);
  
  // quadrature points per face
  int nQ = (int)std::ceil(order+1/2.0);
  nFQNodes = initFaceMap(efToQ, nQ);
  mpi.initDatatype(nFQNodes);
  
}

/** Initialize global quadrature points from solver's reference quadrature points */
void Mesh::setupQuads(const darray& xQV, int nQV) {
  
  globalQuads.realloc(DIM, nQV, nElements);
  for (int iK = 0; iK < nElements; ++iK) {
    for (int iQ = 0; iQ < nQV; ++iQ) {
      for (int l = 0; l < DIM; ++l) {
	globalQuads(l,iQ,iK) = tempMapping(0,l,iK)*xQV(l,iQ)+tempMapping(1,l,iK);
      }
    }
  }
  
}

/**
   Initialize face maps for a given number of nodes.
   Assumes that each face is a side with exactly size1D nodes.
   
   For every element k, face i, face node iFN:
   My node @: soln(efMap(iFN, i), :, k)
   Neighbor's node @: soln(efMap(iFN, eToF(i,k)), :, eToE(i,k))
*/
int Mesh::initFaceMap(iarray& efMap, int size1D) {
  
  efMap.realloc(size1D, N_FACES);
  
  int xOff;
  int yOff;
  
  // -x direction face
  xOff = 0;
  for (int iy = 0; iy < size1D; ++iy) {
    yOff = iy*size1D;
    efMap(iy, 0) = xOff+yOff;
  }
  
  // +x direction face
  xOff = size1D-1;
  for (int iy = 0; iy < size1D; ++iy) {
    yOff = iy*size1D;
    efMap(iy, 1) = xOff+yOff;
  }
  
  // -y direction face
  yOff = 0;
  for (int ix = 0; ix < size1D; ++ix) {
    xOff = ix;
    efMap(ix, 2) = xOff+yOff;
  }
  
  // +y direction face
  yOff = (size1D-1)*size1D;
  for (int ix = 0; ix < size1D; ++ix) {
    xOff = ix;
    efMap(ix, 3) = xOff+yOff;
  }
  
  // Re-organize face nodes so they are accessible by one index
  int tempFNodes = size1D;
  efMap.resize(tempFNodes, N_FACES);
  return tempFNodes;
  
}

std::ostream& operator<<(std::ostream& out, const Mesh& mesh) {
  out << mesh.nVertices << " vertices connected with " << mesh.nElements << " elements." << std::endl;
  return out;
}

