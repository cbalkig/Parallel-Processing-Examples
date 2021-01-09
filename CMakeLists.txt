cmake_minimum_required(VERSION 3.15)
project(MPI C)

set(CMAKE_C_STANDARD 99)

find_package(OpenMP REQUIRED)

add_executable(MPI mpi.c openmp.c summation1.c)
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

target_link_libraries(MPI OpenMP::OpenMP_C mpi openmp)