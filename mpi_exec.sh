#!/bin/bash
mpicc summation1.c -o mpi
mpirun -np 4 ./mpi