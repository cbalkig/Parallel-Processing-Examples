#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#define m 1024

int calc(int a, int c, int x);

int main(int argc, char *argv[]) {
    int a = rand() % m;
    int c = rand() % m;

    int process_count, my_id, err;
    MPI_Status status;
    int root = 0;

    err = MPI_Init(&argc, &argv);
    if (err != 0) {
        printf("Init Error %d.", err);
        exit(-1);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

    int prev_x, x = 134775813;
    if (root != my_id) {
        MPI_Recv(&prev_x, 1, MPI_INT, my_id - 1, 0, MPI_COMM_WORLD, &status);
        printf("Process %d: I've received %d from Process %d.\n", my_id, prev_x, my_id - 1);
    }
    x = calc(a, c, prev_x);
    if (my_id < process_count - 1) {
        MPI_Send(&x, 1, MPI_INT, my_id + 1, 0, MPI_COMM_WORLD);
        printf("Process %d: I've sent %d to Process %d.\n", my_id, x, my_id + 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    printf("Process %d: My initial x: %d\n", my_id, x);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < process_count; j++) {
            x = calc(a, c, x);
        }
        //printf("Process %d: Iteration: %d My Value: %d.\n", my_id, i, x);
    }

    printf("Process %d: My final x: %d\n", my_id, x);

    exit(0);
}

int calc(int a, int c, int x) {
    return (a * x + c) % m;
}
