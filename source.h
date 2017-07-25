#ifndef SOURCE_H__
#define SOURCE_H__

#include <cmath>
#include <iostream>

#include "mesh.h"
#include "rk4.h"
#include "array.h"

class Source {
public:

  enum class Wavelet {
    cos, ricker, rtm, null
  };
  
  // Input parameters for specifying source
  typedef struct {
    Wavelet type;
    int halfSrc;
    int timesteps;
    double dt;
    double maxF;
    darray srcPos;
    darray srcAmps;
  } Params;
  
  Wavelet type;   // type of wavelet
  int halfSrc;    // timesteps when src != 0
  int nt;         // timesteps total
  darray wavelet; // wavelet amplitude in time
  darray weights; // spatial weights
  
  Source();
  Source(const Params& in);
  void init(const Params& in);
  void definePositions(const Params& in, const Mesh& mesh, const darray& xQV);
  
private:
  
  void initCos(double dt, double maxF);
  void initRtm(double dt, double minF, double maxF);
  
  /** Weight in gaussian of which to penalize distance from sources */
  const static double DWEIGHT = 0.4;
  
};


#endif
