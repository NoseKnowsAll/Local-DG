#ifndef SOURCE_H__
#define SOURCE_H__

#include <cmath>
#include <iostream>

#include "mesh.h"
#include "rk4.h"
#include "array.h"

class Source {
public:

  /** Allowable types of wavelets*/
  enum class Wavelet {
    cos, ricker, rtm, spike, null
  };
  
  // Input parameters for specifying source
  typedef struct {
    Wavelet type  = Wavelet::null;
    int timesteps = 1;
    double dt     = 1.0;
    double vsMin  = 0.0;
    double dxMax  = 1.0;
    double maxF   = 0.0; // requested maxF
    darray srcPos;
    darray srcAmps;
  } Params;
  
  Wavelet type;   // type of wavelet
  double maxF;    // maximum frequency of source wavelet
  double t0;      // initial time of source starting
  int halfSrc;    // timesteps until t=0
  int nt;         // timesteps total
  darray wavelet; // wavelet amplitude in time
  darray weights; // spatial weights
  
  Source();
  Source(const Params& in);
  void init(const Params& in);
  void definePositions(const Params& in, const Mesh& mesh);
  
private:
  
  void initCos(double dt, double maxF);
  void initRtm(double dt, double minF, double maxF);
  
  /** Weight in gaussian of which to penalize distance from sources */
  constexpr static double DWEIGHT = 0.4;
  
};


#endif
