#include "mesh.h"

/** Point functionality */
Point::Point() : Point{0.0, 0.0, 0.0} {}

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

Mesh::Mesh(int nx, int ny, int nz) : Mesh{nx, ny, nz, Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 1.0}} {}

Mesh::Mesh(int nx, int ny, int nz, const Point& _botLeft, const Point& _topRight) : Mesh{1, nx, ny, nz, _botLeft, _topRight} {}

/** Main constructor */
Mesh::Mesh(int _order, int nx, int ny, int nz, const Point& _botLeft, const Point& _topRight) :
  nElements{nx*ny*nz},
  nVertices{(nx+1)*(ny+1)*(nz+1)},
  botLeft{_botLeft},
  topRight{_topRight},
  minDX{},
  minDY{},
  minDZ{},
  order{_order},
  nNodes{},
  nFNodes{},
  nFQNodes{},
  globalCoords{},
  vertices{},
  eToV{},
  eToE{},
  normals{},
  eToF{},
  efToN{},
  efToQ{}
{
  
  // Initialize vertices
  vertices.realloc(DIM, nVertices);
  for (int iz = 0; iz <= nz; ++iz) {
    double currZ = iz*(topRight.z-botLeft.z)/nz + botLeft.z;
    for (int iy = 0; iy <= ny; ++iy) {
      double currY = iy*(topRight.y-botLeft.y)/ny + botLeft.y;
      for (int ix = 0; ix <= nx; ++ix) {
	double currX = ix*(topRight.x-botLeft.x)/nx + botLeft.x;
	int vID = ix+iy*(nx+1)+iz*(nx+1)*(ny+1);
	vertices(0, vID) = currX;
	vertices(1, vID) = currY;
	vertices(2, vID) = currZ;
      }
    }
  }
  minDX = (topRight.x - botLeft.x)/nx;
  minDY = (topRight.y - botLeft.y)/ny;
  minDZ = (topRight.z - botLeft.z)/nz;
  
  
  // Initialize elements-to-vertices array
  eToV.realloc(N_VERTICES, nElements);
  for (int iz = 0; iz < nz; ++iz) {
    int zOff1 = (iz  )*(nx+1)*(ny+1);
    int zOff2 = (iz+1)*(nx+1)*(ny+1);
    for (int iy = 0; iy < ny; ++iy) {
      int yOff1 = (iy  )*(nx+1);
      int yOff2 = (iy+1)*(nx+1);
      for (int ix = 0; ix < nx; ++ix) {
	int eIndex = ix + iy*nx + iz*nx*ny;
	int xOff1 = ix;
	int xOff2 = ix+1;
	
	eToV(0, eIndex) = xOff1+yOff1+zOff1;
	eToV(1, eIndex) = xOff2+yOff1+zOff1;
	eToV(2, eIndex) = xOff1+yOff2+zOff1;
	eToV(3, eIndex) = xOff2+yOff2+zOff1;
	eToV(4, eIndex) = xOff1+yOff1+zOff2;
	eToV(5, eIndex) = xOff2+yOff1+zOff2;
	eToV(6, eIndex) = xOff1+yOff2+zOff2;
	eToV(7, eIndex) = xOff2+yOff2+zOff2;
      }
    }
  }
  
  // Initialize element-to-face arrays
  eToE.realloc(N_FACES, nElements);
  normals.realloc(DIM, N_FACES, nElements);
  eToF.realloc(N_FACES, nElements);
  
  // Modulo enforces periodic boundary conditions
  for (int iz = 0; iz < nz; ++iz) {
    int izM = ((iz+nz-1)%nz)*nx*ny;
    int izP = ((iz+nz+1)%nz)*nx*ny;
    int iz0 = iz*nx*ny;
    for (int iy = 0; iy < ny; ++iy) {
      int iyM = ((iy+ny-1)%ny)*nx;
      int iyP = ((iy+ny+1)%ny)*nx;
      int iy0 = iy*nx;
      for (int ix = 0; ix < nx; ++ix) {
	int ixM = (ix+nx-1)%nx;
	int ixP = (ix+nx+1)%nx;
	int ix0 = ix;
	int eIndex = ix + iy0 + iz0;
	
	// Neighbor elements in -x,+x,-y,+y,-z,+z directions stored in faces
	eToE(0, eIndex) = ixM+iy0+iz0;
	eToE(1, eIndex) = ixP+iy0+iz0;
	eToE(2, eIndex) = ix0+iyM+iz0;
	eToE(3, eIndex) = ix0+iyP+iz0;
	eToE(4, eIndex) = ix0+iy0+izM;
	eToE(5, eIndex) = ix0+iy0+izP;
	
	// Face ID of this element's -x face will be neighbor's +x face (same for all dimensions)
	eToF(0, eIndex) = 1;
	eToF(1, eIndex) = 0;
	eToF(2, eIndex) = 3;
	eToF(3, eIndex) = 2;
	eToF(4, eIndex) = 5;
	eToF(5, eIndex) = 4;
	
	// Note that normals is already filled with 0s
	normals(0, 0, eIndex) = -1.0;
	normals(0, 1, eIndex) = 1.0;
	normals(1, 2, eIndex) = -1.0;
	normals(1, 3, eIndex) = 1.0;
	normals(2, 4, eIndex) = -1.0;
	normals(2, 5, eIndex) = 1.0;
      }
    }
  }
  
}

/** Copy constructor */
Mesh::Mesh(const Mesh& other) :
  nElements{other.nElements},
  botLeft{other.botLeft},
  topRight{other.topRight},
  minDX{other.minDX},
  minDY{other.minDY},
  minDZ{other.minDZ},
  nVertices{other.nVertices},
  order{other.order},
  nNodes{other.nNodes},
  nFNodes{other.nFNodes},
  nFQNodes{other.nFQNodes},
  globalCoords{other.globalCoords},
  vertices{other.vertices},
  eToV{other.eToV},
  eToE{other.eToE},
  normals{other.normals},
  eToF{other.eToF},
  efToN{other.efToN},
  efToQ{other.efToQ}
{ }

/** Initialize global nodes from solver's Chebyshev nodes */
void Mesh::setupNodes(const darray& chebyNodes, int _order) {
  
  order = _order;
  nNodes = chebyNodes.size(1)*chebyNodes.size(2)*chebyNodes.size(3);
  darray refNodes{chebyNodes, DIM, nNodes}; 
  
  // Scales and translates Chebyshev nodes into each element
  globalCoords.realloc(DIM, nNodes, nElements);
  for (int k = 0; k < nElements; ++k) {
    darray botLeft{&vertices(0, eToV(0, k)), DIM};
    darray topRight{&vertices(0, eToV(7, k)), DIM};
    
    for (int iN = 0; iN < nNodes; ++iN) {
      for (int l = 0; l < DIM; ++l) {
	// amount in [0,1] to scale lth dimension
	double scale = .5*(refNodes(l,iN)+1.0);
	globalCoords(l,iN,k) = botLeft(l)+scale*(topRight(l)-botLeft(l));
      }
    }
  }
  
  // nodal points per face
  nFNodes = initFaceMap(efToN, order+1);
  
  // quadrature points per face
  int nQ = (int)std::ceil((order+1)/2.0);
  nFQNodes = initFaceMap(efToQ, nQ);

}

/**
   Initialize face maps for a given number of nodes.
   Assumes that each face is a square with exactly size1D nodes per dimension.
   
   For every element k, face i, face node iFN:
   My node @: soln(efMap(iFN, i), :, k)
   Neighbor's node @: soln(efMap(iFN, eToF(i,k)), :, eToE(i,k))
*/
int Mesh::initFaceMap(iarray& efMap, int size1D) {
  
  efMap.realloc(size1D,size1D, N_FACES);
  
  int xOff;
  int yOff;
  int zOff;
  
  // -x direction face
  xOff = 0;
  for (int iz = 0; iz < size1D; ++iz) {
    zOff = iz*size1D*size1D;
    for (int iy = 0; iy < size1D; ++iy) {
      yOff = iy*size1D;
      efMap(iy,iz, 0) = xOff+yOff+zOff;
    }
  }
  
  // +x direction face
  xOff = size1D-1;
  for (int iz = 0; iz < size1D; ++iz) {
    zOff = iz*size1D*size1D;
    for (int iy = 0; iy < size1D; ++iy) {
      yOff = iy*size1D;
      efMap(iy,iz, 1) = xOff+yOff+zOff;
    }
  }
  
  // -y direction face
  yOff = 0;
  for (int iz = 0; iz < size1D; ++iz) {
    zOff = iz*size1D*size1D;
    for (int ix = 0; ix < size1D; ++ix) {
      xOff = ix;
      efMap(ix,iz, 2) = xOff+yOff+zOff;
    }
  }
  
  // +y direction face
  yOff = (size1D-1)*size1D;
  for (int iz = 0; iz < size1D; ++iz) {
    zOff = iz*size1D*size1D;
    for (int ix = 0; ix < size1D; ++ix) {
      xOff = ix;
      efMap(ix,iz, 3) = xOff+yOff+zOff;
    }
  }
  
  // -z direction face
  zOff = 0;
  for (int iy = 0; iy < size1D; ++iy) {
    yOff = iy*size1D;
    for (int ix = 0; ix < size1D; ++ix) {
      xOff = ix;
      efMap(ix,iy, 4) = xOff+yOff+zOff;
    }
  }
  
  // +z direction face
  zOff = (size1D-1)*size1D*size1D;
  for (int iy = 0; iy < size1D; ++iy) {
    yOff = iy*size1D;
    for (int ix = 0; ix < size1D; ++ix) {
      xOff = ix;
      efMap(ix,iy, 5) = xOff+yOff+zOff;
    }
  }
  
  // Re-organize face nodes so they are accessible by one index
  int tempFNodes = size1D*size1D;
  efMap.resize(tempFNodes, N_FACES);
  return tempFNodes;
  
}

std::ostream& operator<<(std::ostream& out, const Mesh& mesh) {
  out << mesh.nVertices << " vertices connected with " << mesh.nElements << " elements." << std::endl;
  return out;
}

