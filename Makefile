CC = mpiicpc
#g++

DEBUG = -O0 -g -traceback -Wcheck -ftrapuv -debug all

CFLAGS  = -O3 -std=c++11 -mkl $(DEBUG)
LFLAGS  = -O3 -std=c++11 -mkl $(DEBUG)
DEFINES = #-Dvariables
HEADERS = solver.h mesh.h dgMath.h io.h MPIUtil.h array.h
OBJS    = driver.o solver.o mesh.o dgMath.o io.o MPIUtil.o
LIBS    = -I.

TARGETS = driver

all:	$(TARGETS)

%.o: 	%.cpp $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $< $(LIBS)

driver:	$(OBJS)
	$(CC) $(LFLAGS) -o $@ $^ $(LIBS)

clean :
	rm -f *.o $(TARGETS)

