#ifndef MPI_UTIL_H__
#define MPI_UTIL_H__

#include <mpi.h>
#include <iostream>
#include <sstream>
#include <string>

#include "array.h"

class MPIUtil {
  
public:
  
  /** Default/Main constructor */
  MPIUtil();
  
  /** Initialize MPI_FACE for use in further MPI sends/recvs */
  void initDatatype(int nodesPerFace);
  
  /* Map MPI faces to the first MPIUtil::N_FACES faces in 3D */
  void initFaces(int meshDim);
  
  /** MPI helper function to print out a string */
  void printString(const std::string& toPrint) const;
  
  /** This MPI rank's number */
  int rank;
  /** Total number of MPI tasks */
  int np;
  /** Number of MPI tasks in each dimension */
  iarray nps;
  /** Array of MPI ranks describing neighbors in -x,+x,-y,+y,-z,+z directions */
  iarray neighbors;
  
  /** DIM array defining this rank's coordinate in cartComm */
  iarray coords;
  
  /** Tag array for each face when on receiving side of communication */
  iarray tags;
  
  /** Cartesian MPI Communicator */
  MPI_Comm cartComm;
  
  /** Mapping a given MPI face to the corresponding mesh face */
  iarray faceMap;
  
  /** Derived datatype describing the nodes on a single element face */
  MPI_Datatype MPI_FACE;
  
  /** Dimension of MPI topology we are modeling */ 
  const static int DIM = 3;
  /** Number of faces per MPI rank */
  const static int N_FACES = 2*DIM;
  /** Root node rank */
  const static int ROOT = 0;
  
};


#endif
