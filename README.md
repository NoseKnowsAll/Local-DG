# Local-DG
Local Discontinuous Galerkin Solver written in C++ and MPI for use on supercomputers such as NERSC's Edison or TACC's Stampede to solve problems involving the Navier-Stokes equations in 3D.

Can be compiled using the MPI Intel compiler (`mpiicpc`) or MPI GCC compiler (`mpic++`).

To install, run `git clone https://github.com/NoseKnowsAll/Navier-Stokes.git`

## Running the Program

Main options that can be changed are available in `driver.cpp` and in the constructor `Solver::Solver()` within `solver.cpp`.

First compile the program:

1) On Edison:
* Ensure `CC` defined as `mpiicpc` in Makefile
* `source init.sh`
* `make`

2) On Stampede:
* Ensure `CC` defined as `mpiicpc` in Makefile
* `make`

3) On Implicit:
* Ensure `CC` defined as `mpic++` in Makefile
* `make`

Once code is successfully compiled, run executable with `./driver`

## Contributing to the project

If you are interested in contributing to the project, please contact Michael Franco at masshellions@gmail.com. At this time, the code is largely research-code, so I don't expect anyone to take me up on this offer. That being said, I welcome any and all feedback/comments on this project at the same email.

## Licensing

Copyright (C) 2017 Franco Computational Solutions

    This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
	    the Free Software Foundation, either version 3 of the License, or
	        (at your option) any later version.

    This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
	    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	        GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.