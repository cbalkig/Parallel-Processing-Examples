#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#define N 500

void calcPrimes(int numbers[N], int number);

int getNumber(int numbers[N], int number);

void printPrimes(int numbers[N]);

int main(int argc, char *argv[]) {
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

    int numbers[N];
    numbers[0] = 0;
    for (int i = 1; i < N; i++) {
        numbers[i] = 1;
    }

    int number = 2;
    int tag = 0;

    int next_process = -1;
    int prev_process = -1;

    if (root == my_id) {
        next_process = my_id + 1;
        prev_process = process_count - 1;
    } else if (process_count - 1 == my_id) {
        next_process = root;
        prev_process = my_id - 1;
    } else {
        next_process = my_id + 1;
        prev_process = my_id - 1;
    }

    if (my_id != root) {
        MPI_Recv(&numbers, N, MPI_INT, prev_process, 0, MPI_COMM_WORLD, &status);
        tag = status.MPI_TAG;
        if (99 != tag) {
            MPI_Recv(&number, 1, MPI_INT, prev_process, 0, MPI_COMM_WORLD, &status);
            printf("Process %d: I've received %d from Process %d.\n", my_id, number, prev_process);
        }
    }

    while (99 != tag) {
        calcPrimes(numbers, number);
        number = getNumber(numbers, number);

        if (-1 != number) {
            MPI_Send(&numbers, N, MPI_INT, next_process, 0, MPI_COMM_WORLD);
            MPI_Send(&number, 1, MPI_INT, next_process, 0, MPI_COMM_WORLD);
            printf("Process %d: I've sent %d to Process %d.\n", my_id, number, next_process);
        } else {
            printPrimes(numbers);
            MPI_Send(&numbers, N, MPI_INT, next_process, 99, MPI_COMM_WORLD);
            break;
        }

        MPI_Recv(&numbers, N, MPI_INT, prev_process, 0, MPI_COMM_WORLD, &status);
        tag = status.MPI_TAG;
        if (99 != tag) {
            MPI_Recv(&number, 1, MPI_INT, prev_process, 0, MPI_COMM_WORLD, &status);
            printf("Process %d: I've received %d from Process %d.\n", my_id, number, prev_process);
        }
    }

    exit(0);
}

void calcPrimes(int numbers[N], int number) {
    for (int i = 0; i < N; i++) {
        if (1 == numbers[i] && (i + 1) > number && (i + 1) % number == 0) {
            numbers[i] = 0;
        }
    }
}

int getNumber(int numbers[N], int number) {
    for (int i = number; i < sqrt(N); i++) {
        if (1 == numbers[i]) {
            return i + 1;
        }
    }

    return -1;
}

void printPrimes(int numbers[N]) {
    printf("The primes numbers are:\n");
    for (int i = 0; i < N; i++) {
        if (1 == numbers[i]) {
            printf("%d\t", i + 1);
        }
    }
    printf("\n");
}