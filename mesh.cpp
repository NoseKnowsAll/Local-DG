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






/** Element functionality */
// Default constructor
Element::Element() {
  
}

Element::Element(const Point& one, const Point& two) : Element{1, one, two} {}

// Two points make up the two opposite corners of cube
// Convenience constructor for structured grids
Element::Element(int _order, const Point& one, const Point& two) :
  corners{},
  dofs{},
  order{_order}
{
  Point p2{one.x, one.y, two.z};
  Point p3{one.x, two.y, one.z};
  Point p4{one.x, two.y, two.z};
  Point p5{two.x, one.y, one.z};
  Point p6{two.x, one.y, two.z};
  Point p7{two.x, two.y, one.z};
  
  corners.resize(N_VERTICES);
  corners.push_back(one);
  corners.push_back(p2);
  corners.push_back(p3);
  corners.push_back(p4);
  corners.push_back(p5);
  corners.push_back(p6);
  corners.push_back(p7);
  corners.push_back(two);
  
  initDoFs();
}

// Constructor for arbitrary grids of order 1
Element::Element(const std::vector<Point>& _corners) : Element{1, _corners} {}

// Actual constructor for arbitrary grids
Element::Element(int _order, const std::vector<Point>& _corners) :
  corners{_corners},
  dofs{},
  order{_order}
{
  initDoFs();
}

// Copy constructor
Element::Element(const Element& other) :
  corners{other.corners},
  dofs{other.dofs},
  order{other.order}
{ }

/** Assignment operator */
Element& Element::operator=(const Element& elem) {
  if (this != &elem) {
    corners = elem.corners;
    dofs = elem.dofs;
    order = elem.order;
  }
  return *this;
  
}

/** After order and corners have been initialized, computes locations of dofs */
void Element::initDoFs() {
  dofs.resize((order+1)*(order+1));
  // TODO: figure out where distribution of points goes
  // Could use Gauss-Legendre-Lobatto zeros,
  // Radau zeros, or Gauss-Legendre zeros
  // They only need to be computed once, and then mapped to each element. Is there a clean way to do so?
  std::cerr << "ERROR: dofs not initialized yet!" << std::endl;
}

/** Allows for Elements to be printed out to command line */
std::ostream& operator<<(std::ostream& out, const Element& elem) {
  for (const Point& p : elem.corners) {
    out << p << std::endl;
  }
  return out;
}

/** Comparison operators */
inline bool operator==(const Element& lhs, const Element& rhs) {
  if (lhs.order != rhs.order)
    return false;

  // Equality means that they have the same corners in some order
  for (const Point& p: lhs.corners) {
    if (std::find(rhs.corners.begin(), rhs.corners.end(), p) == rhs.corners.end())
      return false;
  }
  return true;
}

inline bool operator!=(const Element& lhs, const Element& rhs) {
  return !(lhs == rhs);
}

/** End Element Functionality */








/** Mesh Functionality */
Mesh::Mesh() : Mesh{10,10,10} {}

Mesh::Mesh(int nx, int ny, int nz) : Mesh{nx, ny, nz, Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 1.0}} {}

Mesh::Mesh(int nx, int ny, int nz, const Point& botLeft, const Point& topRight) : Mesh{1, nx, ny, nz, botLeft, topRight} {}

/** Main constructor */
Mesh::Mesh(int _order, int nx, int ny, int nz, const Point& botLeft, const Point& topRight) :
  globalCoords{},
  nElements{nx*ny*nz},
  nVertices{(nx+1)*(ny+1)*(nz+1)},
  order{_order},
  nNodes{},
  nFNodes{},
  vertices{},
  eToV{},
  eToE{},
  normals{},
  eToF{},
  efToN{}
{
  
  // Initialize vertices
  vertices.realloc(DIM, nVertices);
  for (int iz = 0; iz <= nz; ++iz) {
    double currZ = iz*(topRight.z-botLeft.z)/nz + botLeft.z;
    for (int iy = 0; iy <= ny; ++iy) {
      double currY = iy*(topRight.y-botLeft.y)/ny + botLeft.y;
      for (int ix = 0; ix <= nx; ++ix) {
	double currX = ix*(topRight.x-botLeft.x)/nx + botLeft.x;
	vertices(0, ix+iy*(nx+1)+iz*(nx+1)*(ny+1)) = currX;
	vertices(1, ix+iy*(nx+1)+iz*(nx+1)*(ny+1)) = currY;
	vertices(2, ix+iy*(nx+1)+iz*(nx+1)*(ny+1)) = currZ;
      }
    }
  }
  
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
	
	// Uses the fact that normals is initialized with 0s
	normals(0, 0, eIndex) = -1.0;
	normals(0, 1, eIndex) = 1.0;
	normals(1, 2, eIndex) = -1.0;
	normals(1, 3, eIndex) = 1.0;
	normals(2, 4, eIndex) = -1.0;
	normals(2, 5, eIndex) = 1.0;
      }
    }
  }
  
  
  /* Object oriented way of initializing vertices
  for (int iz = 0; iz < nz; ++iz) {
    double botZ = (iz  )*(topRight.z-botLeft.z)/nz + botLeft.z;
    double topZ = (iz+1)*(topRight.z-botLeft.z)/nz + botLeft.z;
    for (int iy = 0; iy < ny; ++iy) {
      double southY = (iy  )*(topRight.y-botLeft.y)/ny + botLeft.y;
      double northY = (iy+1)*(topRight.y-botLeft.y)/ny + botLeft.y;
      for (int ix = 0; ix < nx; ++ix) {
      	double westX = (ix  )*(topRight.x-botLeft.x)/nx + botLeft.x;
	double eastX = (ix+1)*(topRight.x-botLeft.x)/nx + botLeft.x;
	
	// Create element at this location
	Point one{westX, southY, botZ};
	Point two{eastX, northY, topZ};
	Element elem{order, one, two};
	elements.push_back(elem);
	
      }
    }
  }
  */
  
}

/** Copy constructor */
Mesh::Mesh(const Mesh& other) :
  globalCoords{other.globalCoords},
  nElements{other.nElements},
  nVertices{other.nVertices},
  order{other.order},
  nNodes{other.nNodes},
  nFNodes{other.nFNodes},
  vertices{other.vertices},
  eToV{other.eToV},
  eToE{other.eToE},
  normals{other.normals},
  eToF{other.eToF},
  efToN{other.efToN}
{ }

/** Initialize global nodes from solver's Chebyshev nodes */
void Mesh::setupNodes(const darray& chebyNodes, int _order) {
  
  order = _order;
  nNodes = chebyNodes.size(1);
  globalCoords.realloc(DIM, nNodes, nElements);
  // Assumes that nNodes = (order+1)^3 and each face is a square
  nFNodes = (order+1)*(order+1);
  
  for (int k = 0; k < nElements; ++k) {
    darray botLeft{vertices(0, eToV(0, k)), DIM};
    darray topRight{vertices(0, eToV(7, k)), DIM};
    darray diff{topRight - botLeft};
    
    for (int iN = 0; iN < nNodes; ++iN) {
      for (int j = 0; j < DIM; ++j) {
	// amount in [0,1] to scale jth dimension
	double scale = .5*(chebyNodes(j,iN)+1);
	globalCoords(j,iN,k) = botLeft(j)+scale*diff(j);
      }
    }
  }
  
  // TODO: initialize efToN now that we have nodes
  
}

std::ostream& operator<<(std::ostream& out, const Mesh& mesh) {
  out << mesh.nVertices << " vertices connected with " << mesh.nElements << " elements." << std::endl;
  return out;
}

