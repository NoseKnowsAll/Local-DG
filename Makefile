CC = g++

CFLAGS  = -O3 -std=c++11
LFLAGS  = -O3 -std=c++11
DEFINES = #-Dvariables
HEADERS = mesh.h array.h
OBJS    = driver.o mesh.o
LIBS    = -I.

TARGETS = driver

all:	$(TARGETS)

%.o: 	%.cpp $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $< $(LIBS)

driver:	$(OBJS)
	$(CC) $(LFLAGS) -o $@ $^ $(LIBS)

clean :
	rm -f *.o $(TARGETS)

