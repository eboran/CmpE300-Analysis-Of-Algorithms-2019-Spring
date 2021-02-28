# CmpE300-Analysis-Of-Algorithms-2019-Spring

Parallel Programming Project by Using MPI 

* Windows (MPICH2)
gcc -L"C:\Program Files (x86)\MPICH2\lib" -I"C:\Program Files (x86)\MPICH2\include"
mpi_project.c -lmpi -o mpi_project.exe
mpiexec -n NUM_OF_PROCESSORS ./mpi_project.exe

* Unix/Max (OpenMPI)
mpicc mpi_project.c -o mpi_project.o
mpirun -np NUM_OF_PROCESSORS ./mpi_project.o

* Ubuntu
mpicc -g mpi_project.c -o mpi_project
mpiexec -n NUM_OF_PROCESSORS ./mpi_project
