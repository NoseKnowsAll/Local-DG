#include "io.h"

/**
   Clears a space separated file for first time use.
   For use with MATLAB.
*/
bool clearSSVFile(const std::string& filename) {
  std::ofstream outFile(filename, std::ios::out);
  if (outFile.fail()) {
    std::cerr << "ERROR: could not open file " << filename << std::endl;
    return false;
  }
  
  return true;
}

/**
   Outputs array to a txt file using space separated values. 
   Concatenates to file if it exists.
   For use with MATLAB.
*/
bool exportToSSVFile(const std::string& filename, const darray& arr, int dim0, int dim1) {
  
  std::ofstream outFile(filename, std::ios::out | std::ios::app);
  if (outFile.fail()) {
    std::cerr << "ERROR: could not open file " << filename << std::endl;
    return false;
  }
  
  for (int i = 0; i < dim1; ++i) {
    for (int j = 0; j < dim0; ++j) {
      outFile << arr(j,i) << " ";
    }
    outFile << "\n";
  }
  outFile << std::endl;
  
  return true;
  
}


/**
   Clears a file and sets up the X-Y-Z-V headers for first time use.
   For use with Paraview.
*/
bool initXYZVFile(const std::string& filename, const std::string& valuename, int nStates) {
  
  std::ofstream outFile(filename, std::ios::out);
  if (outFile.fail()) {
    std::cerr << "ERROR: could not open file " << filename << std::endl;
    return false;
  }
  
  outFile << "X, Y, Z, ";
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
bool initXYZVFile(const std::string& filename, int timeseries, const std::string& valuename, int nStates) {
  std::ostringstream oss;
  oss << filename << "." << timeseries;
  
  return initXYZVFile(oss.str(), valuename, nStates);
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
  
  // Assumes array is of size (:,nStates;:)
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
