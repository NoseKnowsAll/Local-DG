#ifndef MESH_H__
#define MESH_H__

#ifdef __INTEL_COMPILER
#include <mkl_lapacke.h>
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
#include <cmath>
#include <algorithm>
#include <iostream>
#include <set>
#include <limits>

#include "io.h"
#include "MPIUtil.h"
#include "dgMath.h"
#include "array.h"

/** Singular point in 2D or 3D */
class Point {
public:
  
  double x;
  double y;
  double z;
  
  Point();
  Point(double _x, double _y);
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
  
  /** Boundary conditions */
  enum class Boundary : int {
    free = -1, absorbing = -2
  };
  
  /** Dimension of space we are modeling */ 
  const static int DIM = 2;
  /** Number of faces per element */
  const static int N_FACES = 2*DIM;
  /** Number of vertices defining element */
  const static int N_VERTICES = 1<<DIM; // 2^DIM
  /** Number of vertices defining face */
  const static int N_FVERTICES = 1<<(DIM-1); // 2^(DIM-1)
  
  Mesh();
  Mesh(const MPIUtil& _mpi);
  Mesh(int nx, int ny);
  Mesh(int nx, int ny, const Point& _botLeft, const Point& _topRight);
  Mesh(int nx, int ny, const Point& _botLeft, const Point& _topRight, const MPIUtil& _mpi);
  Mesh(const Mesh& other);
  
  Mesh(const std::string& filename, const MPIUtil& _mpi);
  
  /** Initialize mappings assuming evenly distributed square */
  void defaultSquare(int nx, int ny);
  
  /** Initialize global nodes from bilinear mapping of reference nodes */
  void setupNodes(const darray& InterpTk, int _order);
  
  /** Initialize global quadrature points from bilinear mappings of reference quadrature points */
  void setupQuads(const darray& InterpTkQ, int nQV);
  
  /**
     Initializes absolute value of determinant of Jacobian of mappings
     calculated at the quadrature points xQV
     within the volume stored in Jk and along the faces stored in JkF
  */
  void setupJacobians(int nQV, const darray& xQV, darray& Jk,
		      int nQF, const darray& xQF, darray& JkF) const;
  
  friend std::ostream& operator<<(std::ostream& out, const Mesh& mesh);
  
  /** MPI Utility class */
  MPIUtil mpi;
  /** MPI-local number of elements that this MPI task is responsible for */
  int nElements;
  /** MPI-local number of interior elements */
  int nIElements;
  /** MPI-local number of boundary elements in all directions */
  int nBElements;
  /** MPI-local number of ghost elements in all direction (not including corners outside domain) */
  int nGElements;
  /** MPI-local number of boundary elements in each dimension - one per MPI face */
  iarray mpiNBElems;
  /** MPI-local number of vertices */
  int nVertices;
  /** Global bottom left corner of domain */
  Point botLeft;
  /** Global bottom left corner of domain */
  Point topRight;
  /** Global minimum "dx" across all elements */
  double dxMin;
  /** Global maximum "dx" across all elements */
  double dxMax;
  
  ///////////////////////////////////////
  // Initialized after solver is created
  ///////////////////////////////////////
  /** Order of DG method */
  int order;
  /**
     True coordinates of chebyshev nodes for each element:
     For every element k, node i,
     globalCoords(:, i,k) = 3D location of node i
  */
  darray globalCoords;
  /**
     True coordinates of quadrature points for each element:
     For every element k, quadrature point i,
     globalQuads(:, i,k) = 3D location of quadrature point i
  */
  darray globalQuads;
  ///////////////////////////////////////
  // End of variables initialized after solver is created
  ///////////////////////////////////////
  
  /**
     Global vector of 2D vertices that form corners of elements:
     For every dimension l, vertex i
     vertices(l, i) = x_l of vertex i
  */
  darray vertices;
  
  /**
     element-to-vertex mapping:
     For every element k, vertex i
     eToV(i, k) = MPI-local vertex index of local vertex i
   */
  iarray eToV;
  
  /**
     face-to-vertex mapping:
     For every face j, face vertex i
     fToV(i,f) = vertex index
  */
  iarray fToV;
  
  /**
     boundary element-to-element map:
     For every boundary element k,
     beToE(k) = element number in MPI-local array
  */
  iarray beToE;
  /**
     interior element-to-element map:
     For every interior element k,
     ieToE(k) = element number in MPI-local array
  */
  iarray ieToE;
  
  /**
     MPI boundary element-to-element map:
     For every MPI boundary element k, in MPI face i,
     mpibeToE(k, i) = element number in MPI-local array of neighbor this MPI task controls
  */
  iarray mpibeToE;
  
  /**
     element-to-element neighbor map:
     For every element k, face i
     eToE(i, k) = element number of neighbor on ith face (includes ghost elements)
     3D: -x,+x,-y,+y,-z,+z directions
     If eToE(i,k) < 0, then eToE(i,k) = boundary condition (BC) specified by enum
  */
  iarray eToE;
  /**
     element-to-faces neighbor map:
     For every element k, face i, 
     eToF(i, k) = face number on neighbor's face array
  */
  iarray eToF;
  /**
     normals at each face of each element:
     For every element k, face i
     normals(:,i,k) = outward normal vector of face i
  */
  darray normals;
  /**
     bilinear mapping Tk at each element:
     For every element k, dimension l
     x_l = sum(bilinearMapping(:,l,k)*phi(xi_l,:))
  */
  darray bilinearMapping;
  /**
     temporary mapping Tk at each element:
     For every element k, dimension l
     x_l = tempMapping(0,l,k)*xi_l + tempMapping(1,l,k)
  */
  darray tempMapping;
  
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
