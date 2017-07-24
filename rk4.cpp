#include "rk4.h"

/** Update current u variable based on diagonal Butcher tableau of RK4 */
void rk4::updateCurr(darray& uCurr, const darray& u, const darray& ks, double dt, int istage) {
  
  int dofs = uCurr.size(0);
  int nStates = uCurr.size(1);
  int nElem = uCurr.size(2);
  
  // Update uCurr
  if (istage == 0) {
    for (int iK = 0; iK < nElem; ++iK) {
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iN = 0; iN < dofs; ++iN) {
	  uCurr(iN,iS,iK) = u(iN,iS,iK);
	}
      }
    }
  }
  else {
    for (int iK = 0; iK < nElem; ++iK) {
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iN = 0; iN < dofs; ++iN) {
	  uCurr(iN,iS,iK) = u(iN,iS,iK) + dt*diagA[istage-1]*ks(iN,iS,iK,istage-1);
	}
      }
    }
  }
}

/** Uses RK4 to integrate u forward by dt time */
void rk4::integrateTime(darray& u, const darray& ks, double dt) {
  int dofs = u.size(0);
  int nStates = u.size(1);
  int nElem = u.size(2);
  
  // Update u according to b in RK4 butcher tableau
  for (int istage = 0; istage < nStages; ++istage) {
    for (int iK = 0; iK < nElem; ++iK) {
      for (int iS = 0; iS < nStates; ++iS) {
	for (int iN = 0; iN < dofs; ++iN) {
	  u(iN,iS,iK) += dt*b[istage]*ks(iN,iS,iK,istage);
	}
      }
    }
  }
  
}
