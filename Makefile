CC = mpiicpc
#mpic++
DEBUG := TRUE




ifdef DEBUG
 ifeq ($(CC), mpiicpc)
  DEBUG_FLAGS = -O0 -g -traceback -Wcheck -ftrapuv -debug all -DDEBUG
 else
  DEBUG_FLAGS = -O0 -g -Wall -Wextra -Wno-unused-parameter -DDEBUG
 endif
endif

ifeq ($(CC), mpiicpc)
 LIBS =
 INCLUDE = -I. -I/usr/include
 LIBES   = -L. -L/usr/lib

 CFLAGS = -mkl -std=c++11 ${DEBUG_FLAGS}
 LFLAGS = -mkl -std=c++11 ${DEBUG_FLAGS}
else
 LAPACK = -llapacke -llapack -lblas -lm
 LIBS    = ${LAPACK}
 INCLUDE = -I. -I/usr/include/ -I/home/mfranco/Lapack/lapack-3.7.0/LAPACKE/include
 LIBES   = -L. -L/usr/lib/ -L/usr/lib/libblas/ -L/home/mfranco/Lapack/lapack-3.7.0/

 CFLAGS  = -O3 -std=c++11 ${DEBUG_FLAGS}
 LFLAGS  = -O3 -std=c++11 ${DEBUG_FLAGS}
endif


HEADERS = solver.h mesh.h source.h dgMath.h io.h MPIUtil.h array.h
OBJS    = driver.o solver.o mesh.o source.o dgMath.o io.o MPIUtil.o

TARGETS = driver

all:	$(TARGETS)

%.o: 	%.cpp $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $< $(INCLUDE)

driver:	$(OBJS)
	$(CC) $(LFLAGS) -o $@ $^ $(INCLUDE) $(LIBES) $(LIBS)

clean :
	rm -f *.o $(TARGETS)

