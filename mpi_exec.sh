#!/bin/bash
mpicc mpi.c -o mpi
mpirun -np 4 ./mpi