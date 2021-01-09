#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void printArray(char* log, int size, int array[size], int my_id);

int main(int argc, char *argv[]) {
    int N = 4;
    int array[] = {1, 2, 3, 4};

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

    if (root == my_id) {
        printArray("Initial Array", N, array, my_id);
    }

    int sum = 0;

    if (root == my_id) {
        MPI_Send(&array[my_id], 1, MPI_INT, my_id + 1, 0, MPI_COMM_WORLD);
        printf("Process %d: I've sent %d to Process %d.\n", my_id, array[my_id], my_id + 1);
    }
    else if (my_id == (process_count - 1)) {
        MPI_Recv(&sum, 1, MPI_INT, my_id - 1, 0, MPI_COMM_WORLD, &status);
        sum += array[my_id];
        printf("Process %d: Sum: %d\n", my_id, sum);
    }
    else {
        MPI_Recv(&sum, 1, MPI_INT, my_id - 1, 0, MPI_COMM_WORLD, &status);
        sum += array[my_id];
        MPI_Send(&sum, 1, MPI_INT, my_id + 1, 0, MPI_COMM_WORLD);
        printf("Process %d: I've sent %d to Process %d.\n", my_id, sum, my_id + 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (root == my_id) {
        MPI_Recv(&sum, 1, MPI_INT, my_id + 1, 0, MPI_COMM_WORLD, &status);
        printf("Process %d: Sum: %d\n", my_id, sum);
    }
    else if (my_id == (process_count - 1)) {
        MPI_Send(&sum, 1, MPI_INT, my_id - 1, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Recv(&sum, 1, MPI_INT, my_id + 1, 0, MPI_COMM_WORLD, &status);
        printf("Process %d: Sum: %d\n", my_id, sum);
        MPI_Send(&sum, 1, MPI_INT, my_id - 1, 0, MPI_COMM_WORLD);
    }

    exit(0);
}

void printArray(char *log, int size, int array[size], int my_id) {
    char *s = (char *) malloc(1000 * sizeof(char));
    sprintf(s, "Process %d: ", my_id);
    sprintf(s, "%s\t%s: \t", s, log);
    for (int row = 0; row < size; row++) {
        sprintf(s, "%s%d\t", s, array[row]);
    }
    sprintf(s, "%s\n", s);
    printf(s);
    free(s);
}
