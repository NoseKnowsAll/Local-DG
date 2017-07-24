#ifndef RK4__
#define RK4__

#include "array.h"

// Defines RK4
namespace rk4 {
  
  // Number of stages of Butcher tableau
  const int nStages = 4;
  // Butcher tableau time-fractions
  const double c[4] = {0.0, 0.5, 0.5, 1.0};
  // Butcher tableau weights
  const double b[4] = {1/6.0, 1/3.0, 1/3.0, 1/6.0};
  // explicit & diagonal => store nnz of A as off-diagonal vector
  const double diagA[3] = {0.5, 0.5, 1.0};
  
  /** Update current u variable based on diagonal Butcher tableau of RK4 */
  void updateCurr(darray& uCurr, const darray& u, const darray& ks, double dt, int istage);
  
  /** Uses RK4 to integrate u forward by dt time */
  void integrateTime(darray& u, const darray& ks, double dt);
  
}


#endif
