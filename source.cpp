#include "source.h"

Source::Source() {}

Source::Source(const Params& in) :
  type{in.type},
  maxF{},
  t0{},
  halfSrc{},
  nt{},
  wavelet{},
  weights{}
{
  init(in);
}

void Source::init(const Params& in) {
  
  // Initialize source properties
  type = in.type;
  
  double ptsPerLambdaMin = 4.0;
  maxF = in.vsMin/(ptsPerLambdaMin*in.dxMax);
  
  if (in.maxF && in.maxF > maxF) {
    std::cerr << "ERROR: Requested frequency is too high for model!\n";
    std::cerr << "Can only handle source frequency up to " << maxF << " Hz" << std::endl;
    exit(-1);
  }
  // Use model's actual maxF if no maxF is requested
  if (in.maxF)
    maxF = in.maxF;
  
  std::cout << "Using source with maximum frequency of " << maxF << " Hz" << std::endl;
  
  switch(type) {
  case Wavelet::cos: {
    t0 = -3.0/maxF;
    break;
  }
  case Wavelet::ricker: {
    double fundF = maxF/2.65;
    t0 = -1.0/fundF;
    break;
  }
  case Wavelet::rtm: {
    t0 = -3.0/maxF;
    break;
  }
  case Wavelet::spike: {
    t0 = 0.0;
    break;
  }
  case Wavelet::null: {
    t0 = 0.0;
    break;
  }
  default: {
    std::cerr << "ERROR: Asking for a bad source wavelet!" << std::endl;
    break;
  }
  }
  
  // Allocate memory and initialize wavelet
  halfSrc = static_cast<int>(std::floor(std::abs(t0)/in.dt));
  nt = std::max(in.timesteps, halfSrc+1) + halfSrc;
  wavelet.realloc(rk4::nStages, nt);
  
  switch(type) {
  case Wavelet::cos: {
    initCos(in.dt, maxF);
    break;
  }
  case Wavelet::ricker: {
    std::cerr << "ERROR: Ricker wavelet is not yet programmed!" << std::endl; 
    break;
  }
  case Wavelet::rtm: {
    double minF = 0.0;
    initRtm(in.dt, minF, maxF);
    break;
  }
  case Wavelet::spike: {
    wavelet(0) = 1.0;
    break;
  }
  case Wavelet::null: {
    break;
  }
  default: {
    std::cerr << "ERROR: Asking for a bad source wavelet!" << std::endl;
    break;
  }
  }
  
}

/**
   Initialize cosine source wavelet at frequency 
   supposedly too high to be correctly modeled
*/
void Source::initCos(double dt, double maxF) {
  
  int nsrc = 2*halfSrc+1;
  double freq = 0.9*maxF;
  
  // Initialize tapered wavelet at 0.9 max frequency
  darray totalSource{rk4::nStages, nsrc};
  for (int it = 0; it < nsrc; ++it) {
    for (int is = 0; is < rk4::nStages; ++is) {
      double tOff = it - halfSrc + rk4::c[is];
      double taper = (it < halfSrc ? 0.5+0.5*std::cos(M_PI*tOff/halfSrc) : 1);
      totalSource(is,it) = taper*std::cos(2*M_PI*freq*tOff*dt);
    }
  }
  
  // Copy wavelet into array and zero the tail
  for (int it = 0; it < std::min(nsrc,nt); ++it) {
    for (int is = 0; is < rk4::nStages; ++is) {
      wavelet(is,it) = totalSource(is,it);
    }
  }
  for (int it = std::min(nsrc,nt); it < nt; ++it) {
    for (int is = 0; is < rk4::nStages; ++is) {
      wavelet(is,it) = 0.0;
    }
  }
  
}

/**
   Initializes Ricker wavelet given its peak frequency
*/
// TODO: convert to C++/MKL
/*  subroutine ricker(wavelet,nt,dt,fund)
    
    use dds , only : fft_nrfft5, fft_rcfftm, fft_crfftm, FFT_ESTIMATE
    use ISO_C_BINDING, only : c_loc, c_f_pointer
    
    integer :: ier
    
    integer, intent(in) :: nt
    real, intent(in)    :: dt,fund
    real, intent(inout) :: wavelet(nt)
    
    real, parameter     :: wexp=2.0
    
    integer             :: it_ctr,it,iw,nw, nt_lcl,nw_lcl
    real(KIND=SPR)      :: a,dw,fft_scale
    complex(KIND=SPR)   :: ci
    
    real(KIND=SPR), allocatable, target :: rwork(:)
    complex(KIND=SPR), pointer          :: cwork(:)
    
    ! Define a few local constants
    
    ci = cmplx(0.0,1.0,SPR)
    nt_lcl = fft_nrfft5(nt)
    dw = pi/(dble(nt_lcl*dt))
    nw = nt/2+1
    nw_lcl = nt_lcl/2+1
    fft_scale = 1.0/sqrt(real(nt_lcl))
    it_ctr = nt/2+1
    
    ! Allocate working array
    
    allocate(rwork(nt_lcl+2)) ! cwork(nw_lcl)
    
    ! Make a time domain Gaussian centered at it_ctr
    ! Copy it to a complex vector
    
    a = -(tau*fund)**2 / (2.0*abs(wexp))
    do it = 0, nt_lcl-1
       rwork(it+1)=-exp(a*(abs(it_ctr-it)*dt)**2)
    enddo
    rwork(nt_lcl+1) = 0.0
    rwork(nt_lcl+2) = 0.0
    
    call C_F_POINTER(c_loc(rwork), cwork, [nw_lcl])
    
    ! Transform to the frequency domain
    
    ier = fft_rcfftm(fft_scale,nt_lcl,1,nt_lcl+2,rwork,FFT_ESTIMATE)
    
    ! Take 2.0 derivatives
    ! ( ...not sure why, but w=0 case results in NaNs... )
    
    cwork(1)=cmplx(0.0,0.0)
    cwork(2:) = cwork(2:)*(/( (ci*iw*dw)**wexp, iw=1,nw_lcl-1 )/)
    
    ! Transform back to time domain
    
    ier = fft_crfftm(fft_scale,nt_lcl,1,nt_lcl+2,rwork,FFT_ESTIMATE)
    
    ! Copy result to wavelet and normalize
    
    wavelet(1:nt:2)= real(cwork(1:nw-1))
    wavelet(2:nt:2)=aimag(cwork(1:nw-1))
    
    wavelet(:)=wavelet(:)/maxval(wavelet(:))
    
    ! Free locally allocated space
    
    nullify(cwork)
    deallocate(rwork)
    
  end subroutine ricker
*/


/**
   Initialize wavelet with a zero-phase source function
   created using a slow DFT (typically for use within RTM)
*/
void Source::initRtm(double dt, double minF, double maxF) {
  
  int nsrc = 2*halfSrc+1;
  double t0 = -halfSrc*dt;
  double df = 0.1;
  int nf = static_cast<int>(std::ceil((maxF-minF)/df));
  
  // Loop over frequencies
  darray totalSource{rk4::nStages,nsrc};
  for (int ifreq = 1; ifreq <= nf; ++ifreq) {
    double freq = minF+(ifreq-1)*df;
    double weightF = (0.5+0.5*std::cos(M_PI*(ifreq-nf/2)/(nf/2+1)));
    weightF = std::pow(weightF, 1.0/3.0);
    
    // Add frequencies to source wavelet
    for (int it = 0; it < nsrc; ++it) {
      for (int is = 0; is < rk4::nStages; ++is) {
	double tOff = it - halfSrc + rk4::c[is];
	totalSource(is,it) += std::cos(2*M_PI*tOff*dt)*weightF;
      }
    }
  }
  
  // Apply a time-domain taper
  for (int it = 0; it < nsrc; ++it) {
    for (int is = 0; is < rk4::nStages; ++is) {
      double tOff = it - halfSrc + rk4::c[is];
      double taper = 0.5+0.5*std::cos(M_PI*tOff/(halfSrc+1));
      totalSource(is,it) *= taper;
    }
  }
  // Normalize
  double maxval = infnorm(totalSource);
  for (int it = 0; it < nsrc; ++it) {
    for (int is = 0; is < rk4::nStages; ++is) {
      totalSource(is,it) /= maxval;
    }
  }
  
  // Copy wavelet into array and zero the tail
  for (int it = 0; it < std::min(nsrc,nt); ++it) {
    for (int is = 0; is < rk4::nStages; ++is) {
      wavelet(is,it) = totalSource(is,it);
    }
  }
  for (int it = std::min(nsrc,nt); it < nt; ++it) {
    for (int is = 0; is < rk4::nStages; ++is) {
      wavelet(is,it) = 0.0;
    }
  }
  
}

/** Initialize source spatial weights based off of input source positions */
void Source::definePositions(const Params& in, const Mesh& mesh) {
  
  int nsrcs = in.srcPos.size(1);
  int nQV = mesh.globalQuads.size(1);
  
  weights.realloc(nQV, mesh.nElements);
  
  for (int i = 0; i < nsrcs; ++i) {
    for (int iK = 0; iK < mesh.nElements; ++iK) {
      for (int iQ = 0; iQ < nQV; ++iQ) {
	
	// Compute distance from quadrature point to source location
	double dist = 0.0;
	for (int l = 0; l < Mesh::DIM; ++l) {
	  dist += std::pow(mesh.globalQuads(l,iQ,iK) - in.srcPos(l,i),2.0);
	}
	
	// Create Gaussian around each source evaluated at each quadrature point
	double value = in.srcAmps(i)*std::exp(-DWEIGHT*dist);
	weights(iQ, iK) += (value > 1e-16 ? value : 0.0);
	
      }
    }
  }
  
}
