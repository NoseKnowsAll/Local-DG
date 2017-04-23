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
bool initXYZVFile(const std::string& filename, const std::string& valuename) {
  
  std::ofstream outFile(filename, std::ios::out);
  if (outFile.fail()) {
    std::cerr << "ERROR: could not open file " << filename << std::endl;
    return false;
  }
  
  outFile << "X, Y, Z, " << valuename << std::endl;
  
  return true;
  
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
  
  // Total size of array compressed into one column
  long long totalSize = 1;
  for (int j = 1; j < globalCoords.ndim(); ++j) {
    totalSize *= globalCoords.size(j);
  }
  
  for (long long j = 0; j < totalSize; ++j) {
    for (int l = 0; l < globalCoords.size(0); ++l) {
      outFile << globalCoords(l, j) << ", ";
    }
    // Assumes array is of same size as globalCoords(1,2:end)
    outFile << arr(j) << "\n";
  }
  return true;
}
