#ifndef OUTPUT_H__
#define OUTPUT_H__

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <mpi.h>

#include "array.h"

/**
   Clears a space separated file for first time use.
   For use with MATLAB.
*/
bool clearSSVFile(const std::string& filename);

/**
   Outputs array to a txt file using space separated values. 
   Concatenates to file if it exists.
   For use with MATLAB.
*/
bool exportToSSVFile(const std::string& filename, const darray& arr, int dim0, int dim1);



/**
   Clears a file and sets up the X-Y-Z-V headers for first time use.
   For use with Paraview.
*/
bool initXYZVFile(const std::string& filename, const std::string& valuename, int nStates);

/**
   Clears a file and sets up the X-Y-Z-V headers for first time use.
   For use with Paraview in a time series output.
*/
bool initXYZVFile(const std::string& filename, int timeseries, const std::string& valuename, int nStates);

/**
   Outputs array to X-Y-Z-V file using global coordinates and value itself. 
   Concatenates to file if it exists.
   For use with Paraview.
*/
bool exportToXYZVFile(const std::string& filename, const darray& globalCoords, const darray& arr);

/**
   Outputs array to X-Y-Z-V file using global coordinates and value itself. 
   Concatenates to file if it exists.
   For use with Paraview in a time series output.
*/
bool exportToXYZVFile(const std::string& filename, int timeseries, const darray& globalCoords, const darray& arr);


#endif
