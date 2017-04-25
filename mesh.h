#ifndef MESH_H__
#define MESH_H__

#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

#include "array.h"


/** Singular point in 3D */
class Point {
public:
  // initialize to 0
  double x;
  double y;
  double z;

  Point();
  Point(double _x, double _y, double _z);
  Point(const Point& other);
  
  // Assignment operator
  Point& operator=(const Point& p);
  
  // Functions
  double dist(const Point& other) const;

};

// Operator overloads
std::ostream& operator<<(std::ostream& out, const Point& p);
inline bool operator==(const Point& lhs, const Point& rhs);
inline bool operator!=(const Point& lhs, const Point& rhs);





/** Mesh in 3D */
class Mesh {
public:
  
  /** Dimension of space we are modeling */ 
  const static int DIM = 3;
  /** Number of faces per element */
  const static int N_FACES = 2*DIM;
  /** Number of vertices defining element */
  const static int N_VERTICES = 8; // 2^DIM
  
  
  Mesh();
  Mesh(int nx, int ny, int nz);
  Mesh(int nx, int ny, int nz, const Point& _botLeft, const Point& _topRight);
  Mesh(int _order, int nx, int ny, int nz, const Point& _botLeft, const Point& _topRight);
  Mesh(const Mesh& other);
  
  /** Initialize global nodes from solver's Chebyshev nodes */
  void setupNodes(const darray& chebyNodes, int _order);
  
  friend std::ostream& operator<<(std::ostream& out, const Mesh& mesh);
  
  
  /** Global number of elements */
  int nElements;
  /** Global number of vertices */
  int nVertices;
  /** Global bottom left corner of domain */
  Point botLeft;
  /** Global bottom left corner of domain */
  Point topRight;
  /** Global minimum dx of one element */
  double minDX;
  /** Global minimum dy of one element */
  double minDY;
  /** Global minimum dz of one element */
  double minDZ;
  
  ///////////////////////////////////////
  // Initialized after solver is created
  ///////////////////////////////////////
  /** Order of DG method */
  int order;
  /** Local number of DG nodes per element */
  int nNodes;
  /** Local number of DG nodes per face of an element */
  int nFNodes;
  /** Local number of DG quadrature points per face of an element */
  int nFQNodes;
  /**
     True coordinates of chebyshev nodes for each element:
     For every element k, node i,
     globalCoords(:, i,k) = 3D location of node i
  */
  darray globalCoords;
  ///////////////////////////////////////
  // End of variables initialized after solver is created
  ///////////////////////////////////////
  
  /** Global vector of 3D vertices that form corners of elements */
  darray vertices;
  
  /**
     element-to-vertex mapping:
     For every element k, vertex i
     eToV(i, k) = global vertex index of local vertex i
   */
  iarray eToV;
  
  /**
     element-to-element neighbor map:
     For every element k, face i
     eToE(i, k) = element number of neighbor in ith direction
     3D: -x,+x,-y,+y,-z,+z directions
  */
  iarray eToE;
  /**
     normals at each face of each element:
     For every element k, face i
     normals(:,i,k) = outward normal vector of face i
  */
  darray normals;
  /**
     element-to-faces neighbor map:
     For every element k, face i, 
     eToF(i, k) = face number on neighbor's face array
  */
  iarray eToF;
  
  /**
     element-face-to-node map:
     For every face i (same across all elements), 
     efToN(:, i) = local node IDs of nodes on face i
  */
  iarray efToN;
  
  /**
     element-face-to-quadrature-point map:
     For every face i (same across all elements), 
     efToQ(:, i) = local quad IDs of quadrature points on face i
  */
  iarray efToQ;

private:

  /**
     Initialize face maps for a given number of nodes.
     Assumes that each face is a square with exactly size1D nodes per dimension.
     
     For every element k, face i, face node iFN:
     My node @: soln(efMap(iFN, i), :, k)
     Neighbor's node @: soln(efMap(iFN, eToF(i,k)), :, eToE(i,k))
  */
  int initFaceMap(iarray& efMap, int size1D);
  
};

#endif
