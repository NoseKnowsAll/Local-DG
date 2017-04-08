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




/** Singular 3D element (cube) */
class Element {
public:
  // array of Points defining boundary of cube
  std::vector<Point> corners;
  // array of Points defining locations of DoF in cube
  std::vector<Point> dofs;
  // order of polynomial to be evaluated on this element
  int order;
  const static int N_FACES = 6;
  const static int N_VERTICES = 8;
  
  Element();
  Element(const Point& one, const Point& two);
  Element(int _order, const Point& one, const Point& two);
  Element(const std::vector<Point>& _corners);
  Element(int _order, const std::vector<Point>& _corners);
  Element(const Element& other);
  
  // Assignment operator
  Element& operator=(const Element& elem);
  
  // Functions
  //bool contains(const Point& p) const;

  friend std::ostream& operator<<(std::ostream& out, const Element& elem);

private:
  void initDoFs();

};

// Operator overloads
inline bool operator==(const Element& lhs, const Element& rhs);
inline bool operator!=(const Element& lhs, const Element& rhs);





class Mesh {
public:
  
  ///////////////////////////////////////
  // Initialized after solver is created
  ///////////////////////////////////////
  /**
     True coordinates of chebyshev nodes for each element:
     For every element k, node i,
     globalCoords(:, i,k) = 3D location of node i
  */
  darray globalCoords;
  
  /** Holds all 3D coordinates of each element's vertices */
  // TODO: should we index into globalCoords? Or copy data for caching purposes?
  //darray faceCoords;
  
  /** Dimension of space we are modeling */ 
  const static int DIM = 3;
  /** Number of faces per element */
  const static int N_FACES = 2*DIM;
  /** Number of vertices defining element */
  const static int N_VERTICES = 8; // 2^DIM
  
  
  Mesh();
  Mesh(int nx, int ny, int nz);
  Mesh(int nx, int ny, int nz, const Point& botLeft, const Point& topRight);
  Mesh(int _order, int nx, int ny, int nz, const Point& botLeft, const Point& topRight);
  Mesh(const Mesh& other);
  
  void setupNodes(const darray& chebyNodes, int _order);
  
  friend std::ostream& operator<<(std::ostream& out, const Mesh& mesh);
  
private:
  
  /** Global number of elements */
  int nElements;
  /** Global number of vertices */
  int nVertices;
  
  ///////////////////////////////////////
  // Initialized after solver is created
  ///////////////////////////////////////
  /** Order of DG method */
  int order;
  /** Local number of DG nodes per element */
  int nNodes;
  /** Local number of DG nodes per face of an element */
  int nFNodes;
  
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
     For every element k, face i, 
     efToN(:, i,k) = local node IDs of nodes on face i
  */
  iarray efToN;
  
};

#endif
