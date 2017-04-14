CC = icpc
#g++

CFLAGS  = -O3 -std=c++11 -mkl
LFLAGS  = -O3 -std=c++11 -mkl
DEFINES = #-Dvariables
HEADERS = solver.h mesh.h dgMath.h array.h
OBJS    = driver.o solver.o mesh.o dgMath.o
LIBS    = -I.

TARGETS = driver

all:	$(TARGETS)

%.o: 	%.cpp $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $< $(LIBS)

driver:	$(OBJS)
	$(CC) $(LFLAGS) -o $@ $^ $(LIBS)

clean :
	rm -f *.o $(TARGETS)

