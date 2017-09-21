#include "io.h"


/**
   Reads an input .msh file (output from Gmsh) and initializes several key parameters of mesh.h
   Returns true or false based off of success of reading file
*/
bool readMesh(const std::string& filename, int dim, int n_vertices,
	      int& nVertices, int& nElements, darray& vertices,
	      iarray& eToV, iarray& periodicity) {

  std::ifstream mshFile(filename, std::ios::in);
  if (mshFile.fail()) {
    std::cerr << "ERROR: could not open mesh " << filename << std::endl;
    return false;
  }

  // Skip header until reached "$Nodes" line
  std::string nextLine;
  while(std::getline(mshFile, nextLine)) {
    if (nextLine == "$Nodes")
      break;
  }

  // Read in vertices information
  mshFile >> nVertices;
  vertices.realloc(dim, nVertices);
  double dummy;
  int vID;
  for (int i = 0; i < nVertices; ++i) {
    mshFile >> vID >> vertices(0,i) >> vertices(1,i);
    if (dim == 3)
      mshFile >> vertices(2,i);
    else
      mshFile >> dummy;
    
    if (vID != i+1) {
      std::cerr << "FATAL WARNING: This solver currently only accepts meshes with sequential vertex IDs!" << std::endl;
      return false;
    }
  }
  
  // Skip more header information
  while(std::getline(mshFile, nextLine)) {
    if (nextLine == "$Elements")
      break;
  }
  
  // Read in element information
  mshFile >> nElements;
  eToV.realloc(n_vertices, nElements);
  int eID, eType;
  int ntags, dummytag1, dummytag2;
  for (int k = 0; k < nElements; ++k) {
    mshFile >> eID >> eType >> ntags >> dummytag1 >> dummytag2;
    if (ntags != 2) {
      std::cerr << "FATAL WARNING: This solver currently only accepts elements with 2 tags!" << std::endl;
      return false;
    }
    if (eID != k+1) {
      std::cerr << "FATAL WARNING: This solver currently only accepts meshes with sequential element IDs!" << std::endl;
      return false;
    }
    
    // Read in the actual vertices that this element connects
    for (int i = 0; i < n_vertices; ++i) {
      mshFile >> eToV(i,k);
      eToV(i,k) -= 1; // 1-indexing from Gmsh
    }
  }
  
  // Skip more header information
  while(std::getline(mshFile, nextLine)) {
    if (nextLine == "$Periodic")
      break;
  }
  
  // Read in periodic information
  periodicity.realloc(nVertices);
  for (int i = 0; i < nVertices; ++i) {
    periodicity(i) = i;
  }
  
  int nPeriodic, nNodes;
  int dimension, slave, master;
  
  mshFile >> nPeriodic;
  for (int i = 0; i < nPeriodic; ++i) {
    mshFile >> dimension >> slave >> master;
    
    mshFile >> nNodes;
    switch(dimension) {
    case 0: {
      // combining end vertices
      for (int j = 0; j < nNodes; ++j) {
	mshFile >> slave >> master;
	periodicity(slave-1) = master-1;
      }
      break;
    }
    case 1: {
      // combining vertices on edges
      for (int j = 0; j < nNodes; ++j) {
	mshFile >> slave >> master;
	periodicity(slave-1) = master-1;
      }
      break;
    }
    case 2: {
      // combining vertices on faces
      std::cout << "ERROR: I have not programmed periodic faces!" << std::endl;
      for (int j = 0; j < nNodes; ++j) { // TODO: Not yet tried in 3D - is this correct?
	mshFile >> slave >> master;
	periodicity(slave-1) = master-1;
      }
      return false;
      break;
    }
    default: {
      std::cerr << "ERROR: incorrect periodic information!" << std::endl;
      return false;
    }
    }
    
  }
  return true;
  
}

/**
   Reads in the material properties from files
   Also initializes origins and deltas for use throughout program
*/
bool readProps(darray& vp, darray& vs, darray& rhoIn, darray& origins, darray& deltas) {
  return false;
  // TODO: not currently developed!
}

/**
   Clears a file and sets up the X-Y-Z-V headers for first time use.
   For use with Paraview.
*/
bool initXYZVFile(const std::string& filename, int dim, const std::string& valuename, int nStates) {
  
  std::ofstream outFile(filename, std::ios::out);
  if (outFile.fail()) {
    std::cerr << "ERROR: could not open file " << filename << std::endl;
    return false;
  }

  switch (dim) {
  case 1: {
    outFile << "X, ";
    break;
  }
  case 2: {
    outFile << "X, Y, ";
    break;
  }
  case 3: {
    outFile << "X, Y, Z, ";
    break;
  }
  default: {
    outFile << "X3D, Y3D, Z3D, ";
    break;
  }
  }
  
  for (int iS = 0; iS < nStates-1; ++iS) {
    outFile << valuename << iS << ", ";
  }
  outFile << valuename << nStates-1 << std::endl;
  
  return true;
  
}

/**
   Clears a file and sets up the X-Y-Z-V headers for first time use.
   For use with Paraview in a time series output.
*/
bool initXYZVFile(const std::string& filename, int dim, int timeseries, const std::string& valuename, int nStates) {
  std::ostringstream oss;
  oss << filename << "." << timeseries;
  
  return initXYZVFile(oss.str(), dim, valuename, nStates);
}

/**
   Outputs array to X-Y-Z-V file using global coordinates and value itself. 
   Concatenates to file if it exists.
   For use with Paraview.
*/
bool exportToXYZVFile(const std::string& filename, const darray& globalCoords, const darray& arr) {
  
  std::ofstream outFile(filename, std::ios::out | std::ios::app);
  if (outFile.fail()) {
    std::cerr << "ERROR: could not open file " << filename << std::endl;
    return false;
  }
  
  // Assumes array is of size (:,nStates,:)
  int dofs = arr.size(0);
  int nStates = arr.size(1);
  long long totalSize = 1;
  for (int j = 2; j < arr.ndim(); ++j) {
    totalSize *= arr.size(j);
  }
  
  for (long long j = 0; j < totalSize; ++j) {
    for (int i = 0; i < dofs; ++i) {
      
      for (int l = 0; l < globalCoords.size(0); ++l) {
	outFile << globalCoords(l, i, j) << ", ";
      }
      
      for (int iS = 0; iS < nStates-1; ++iS) {
	outFile << arr(i, iS, j) << ", ";
      }
      outFile << arr(i, nStates-1, j) << std::endl;
    }
  }
  return true;
}

/**
   Outputs array to X-Y-Z-V file using global coordinates and value itself. 
   Concatenates to file if it exists.
   For use with Paraview in a time series output.
*/
bool exportToXYZVFile(const std::string& filename, int timeseries, const darray& globalCoords, const darray& arr) {
  std::ostringstream oss;
  oss << filename << "." << timeseries;
  
  int np, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int localSuccess = 0;
  int globalSuccess = 0;
  
  for (int ir = 0; ir < np; ++ir) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == ir) {
      localSuccess = (exportToXYZVFile(oss.str(), globalCoords, arr) ? 1 : 0);
    }
  }
  
  MPI_Allreduce(&localSuccess, &globalSuccess, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
  return (globalSuccess > 0 ? true : false);
}
