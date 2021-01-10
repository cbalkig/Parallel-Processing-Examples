#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define N 8

sort(int size, int numbers[size]);

merge(int size, int numbers[size], int peer_numbers[size], int merged_array[size * 2]);

split(int size, int numbers[size], int *merged_array, int selector);

int main(int argc, char *argv[]) {
    int numbers[N] = {3, 5, 2, 6, 8, 4, 1, 7};

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

    if (my_id == root) {
        printf("Numbers: \t");
        for (int i = 0; i < N; i++) {
            printf("%d\t", numbers[i]);
        }
        printf("\n");
    }

    int array_size = N / process_count;
    int my_number[array_size];
    int peer_number[array_size];
    MPI_Scatter(&numbers, array_size, MPI_INT, &my_number, array_size, MPI_INT, root, MPI_COMM_WORLD);

    sort(array_size, my_number);
    for (int i = 0; i < 3; i++) {
        int selector = -1;
        if (i % 2 == 0) {
            if (my_id % 2 == 0) {
                selector = 0;
                MPI_Send(&my_number, array_size, MPI_INT, my_id + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&peer_number, array_size, MPI_INT, my_id + 1, 0, MPI_COMM_WORLD, &status);
            } else {
                selector = 1;
                MPI_Recv(&peer_number, array_size, MPI_INT, my_id - 1, 0, MPI_COMM_WORLD, &status);
                MPI_Send(&my_number, array_size, MPI_INT, my_id - 1, 0, MPI_COMM_WORLD);
            }
        } else {
            if (my_id % 2 != 0) {
                selector = 0;
                if (my_id + 1 >= process_count) {
                    continue;
                }
                MPI_Send(&my_number, array_size, MPI_INT, my_id + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&peer_number, array_size, MPI_INT, my_id + 1, 0, MPI_COMM_WORLD, &status);
            } else {
                selector = 1;
                if (my_id - 1 < 0) {
                    continue;
                }
                MPI_Recv(&peer_number, array_size, MPI_INT, my_id - 1, 0, MPI_COMM_WORLD, &status);
                MPI_Send(&my_number, array_size, MPI_INT, my_id - 1, 0, MPI_COMM_WORLD);
            }
        }

        int merged_array[array_size * 2];
        merge(array_size, my_number, peer_number, merged_array);
        split(array_size, my_number, merged_array, selector);
        printf("Iteration %d - Process %d - My numbers:\t", i, my_id);
        for (int j = 0; j < array_size; j++) {
            printf("%d\t", my_number[j]);
        }
        printf("\n");
    }

    MPI_Gather(&my_number, array_size, MPI_INT, &numbers, array_size, MPI_INT, root, MPI_COMM_WORLD);

    if (my_id == root) {
        printf("Numbers: \t");
        for (int i = 0; i < N; i++) {
            printf("%d\t", numbers[i]);
        }
        printf("\n");
    }

    exit(0);
}

sort(int size, int numbers[size]) {
    int i, j, min_idx;
    for (i = 0; i < size - 1; i++) {
        min_idx = i;
        for (j = i + 1; j < size; j++)
            if (numbers[j] < numbers[min_idx])
                min_idx = j;
        int temp = numbers[min_idx];
        numbers[min_idx] = numbers[i];
        numbers[i] = temp;
    }
}

merge(int size, int numbers[size], int peer_numbers[size], int merged_array[size]) {
    for (int i = 0; i < size; i++) {
        merged_array[i] = numbers[i];
    }

    for (int i = 0; i < size; i++) {
        merged_array[size + i] = peer_numbers[i];
    }

    sort(size * 2, merged_array);
}

split(int size, int numbers[size], int merged_array[size * 2], int selector) {
    if (selector == 0) {
        for (int i = 0; i < size; i++) {
            numbers[i] = merged_array[i];
        }
    } else {
        for (int i = 0; i < size; i++) {
            numbers[i] = merged_array[size + i];
        }
    }
}