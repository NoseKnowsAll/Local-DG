#ifndef OUTPUT_H__
#define OUTPUT_H__

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include <mpi.h>

#ifdef HDF5
#include <H5Cpp.h>
#include <hdf5_hl.h>
#endif

#include "array.h"


/**
   Reads an input .msh file (output from Gmsh) and initializes several key parameters of mesh.h
*/
bool readMesh(const std::string& filename, int dim, int n_vertices,
	      int& nVertices, int& nElements, darray& vertices,
	      iarray& eToV, iarray& periodicity);

/**
   Reads in the material properties from files
   Also initializes origins and deltas for use throughout program
*/
bool readProps(const std::string& vpFile, const std::string& vsFile, 
	       const std::string& rhoFile, darray& vp, darray& vs, darray& rho,
	       darray& origins, darray& deltas);

/**
   Clears a file and sets up the X-Y-Z-V headers for first time use.
   For use with Paraview.
*/
bool initXYZVFile(const std::string& filename, int dim, const std::string& valuename, int nStates);

/**
   Clears a file and sets up the X-Y-Z-V headers for first time use.
   For use with Paraview in a time series output.
*/
bool initXYZVFile(const std::string& filename, int dim, int timeseries, const std::string& valuename, int nStates);

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
