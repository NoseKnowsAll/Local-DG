#ifndef MESH_H__
#define MESH_H__

#include <cmath>
#include <algorithm>
#include <iostream>

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
  
  //std::vector<Element> elements;
  /** Holds all 3D coordinates of each element's DoFs */
  darray coords;
  /** Holds all 3D coordinates of each element's vertices */
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
  
  friend std::ostream& operator<<(std::ostream& out, const Mesh& mesh);
  
private:
  int nElements;
  int nVertices;
  int order;
  
  /** Holds all of the 3D vertices that form corners of elements */
  darray vertices;
  /** element-to-vertex mapping */
  iarray eToV;
  /** element-to-element neighbor map */
  iarray eToE;
  /** normals at each face of each element */
  darray normals;
  /** element-to-faces neighbor map */
  iarray eToF;
  
};

#endif
