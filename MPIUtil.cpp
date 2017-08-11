#include "MPIUtil.h"

/** Default/Main constructor */
MPIUtil::MPIUtil() {
  
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  nps.realloc(DIM);
  for (int l = 0; l < DIM; ++l) {
    nps[l] = 0; // TODO: Must be 0 before sending to MPI_Dims_create
  }
  MPI_Dims_create(np, DIM, nps.data());
  
  int periods[DIM];
  for (int l = 0; l < DIM; ++l) {
    periods[l] = 1;
  }
  MPI_Cart_create(MPI_COMM_WORLD, DIM, nps.data(), periods, 1, &cartComm);
  
  coords.realloc(DIM);
  MPI_Cart_coords(cartComm, rank, DIM, coords.data());
  
  neighbors.realloc(N_FACES);
  for (int l = 0; l < DIM; ++l) {
    // Gives the address of where to send and receive MPI calls
    MPI_Cart_shift(cartComm, l, 1, &neighbors(2*l), &neighbors(2*l+1));
    
  }
  
  tags.realloc(N_FACES);
  for (int iF = 0; iF < N_FACES; ++iF) {
    tags(iF) = ((iF % 2) == 0 ? iF+1 : iF-1);
  }
  
}

/** Initialize MPI_FACE for use in further MPI sends/recvs */
void MPIUtil::initDatatype(int nodesPerFace) {
  
  MPI_Type_contiguous(nodesPerFace, MPI_DOUBLE, &MPI_FACE);
  MPI_Type_commit(&MPI_FACE);
  
}

/* Map MPI faces to the first MPIUtil::N_FACES faces in 3D */
void MPIUtil::initFaces(int meshDim) {
  faceMap.realloc(N_FACES);
  for (int l = 0; l < N_FACES; ++l) {
    faceMap(l) = meshDim - N_FACES + l;
  }
}

/** MPI helper function to print out a string */
void MPIUtil::printString(const std::string& toPrint) const {
  
  for (int ir = 0; ir < np; ++ir) {
    MPI_Barrier(cartComm);
    if (rank == ir) {
      std::cout << rank << ": " << toPrint << std::endl;
    }
  }
  MPI_Barrier(cartComm);
  
}

/** MPI helper function to gracefully exit all ranks */
void MPIUtil::exit(int err) const {
  
  if (np > 1) {
    MPI_Barrier(cartComm);
  }
  
  MPI_Finalize();
  std::exit(err);
  
}
