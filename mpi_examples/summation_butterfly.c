#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

    if (my_id % 2 == 0) {
        MPI_Recv(&sum, 1, MPI_INT, my_id + 1, 0, MPI_COMM_WORLD, &status);
        MPI_Send(&array[my_id], 1, MPI_INT, my_id + 1, 0, MPI_COMM_WORLD);
        printf("Iteration %d - Process %d: I've sent %d to Process %d.\n", 0, my_id, array[my_id], my_id + 1);
        sum += array[my_id];
        printf("Iteration %d - Process %d: Sum: %d\n", 0, my_id, sum);
    }
    else {
        MPI_Send(&array[my_id], 1, MPI_INT, my_id - 1, 0, MPI_COMM_WORLD);
        printf("Iteration %d - Process %d: I've sent %d to Process %d.\n", 0, my_id, array[my_id], my_id - 1);
        MPI_Recv(&sum, 1, MPI_INT, my_id - 1, 0, MPI_COMM_WORLD, &status);
        sum += array[my_id];
        printf("Iteration %d - Process %d: Sum: %d\n", 0, my_id, sum);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    array[my_id] = sum;
    for (int i = 1; i < sqrt(N); i++) {
        int mod = pow(2, i);
        int stride = pow(2, i);

        if (my_id / mod == 0) {
            MPI_Recv(&sum, 1, MPI_INT, my_id + stride, 0, MPI_COMM_WORLD, &status);
            MPI_Send(&array[my_id], 1, MPI_INT, my_id + stride, 0, MPI_COMM_WORLD);
            printf("Iteration %d - Process %d: I've sent %d to Process %d.\n", i, my_id, array[my_id], my_id + stride);
            sum += array[my_id];
            array[my_id] = sum;
            printf("Iteration %d - Process %d: Sum: %d\n", i, my_id, sum);
        }
        else {
            MPI_Send(&array[my_id], 1, MPI_INT, my_id - stride, 0, MPI_COMM_WORLD);
            printf("Iteration %d - Process %d: I've sent %d to Process %d.\n", i, my_id, array[my_id], my_id - stride);
            MPI_Recv(&sum, 1, MPI_INT, my_id - stride, 0, MPI_COMM_WORLD, &status);
            sum += array[my_id];
            array[my_id] = sum;
            printf("Iteration %d - Process %d: Sum: %d\n", i, my_id, sum);
        }

        MPI_Barrier(MPI_COMM_WORLD);
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
