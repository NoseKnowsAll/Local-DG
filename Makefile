CC = mpic++

DEBUG = -O0 -g -traceback -Wcheck -ftrapuv -debug all
LAPACK = -llapacke -llapack -lblas -lm

CFLAGS  = -O3 -std=c++11
LFLAGS  = -O3 -std=c++11
#${DEBUG} -mkl
DEFINES = #-Dvariables
HEADERS = solver.h mesh.h dgMath.h io.h MPIUtil.h array.h
OBJS    = driver.o solver.o mesh.o dgMath.o io.o MPIUtil.o

LIBS    = ${LAPACK}
INCLUDE = -I. -I/usr/include/ -I/home/mfranco/Lapack/lapack-3.7.0/LAPACKE/include
LIBES   = -L. -L/usr/lib/ -L/usr/lib/libblas/ -L/home/mfranco/Lapack/lapack-3.7.0/

TARGETS = driver

all:	$(TARGETS)

%.o: 	%.cpp $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $< $(INCLUDE)

driver:	$(OBJS)
	$(CC) $(LFLAGS) -o $@ $^ $(INCLUDE) $(LIBES) $(LIBS)

clean :
	rm -f *.o $(TARGETS)

