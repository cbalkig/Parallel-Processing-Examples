#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define N 4

int main(int argc, char *argv[]) {
    int vector[N] = {1, 2, 3, 4};
    int matrix[N][N] = {{1, 1, 1, 1},
                        {2, 2, 2, 2},
                        {3, 3, 3, 3},
                        {4, 4, 4, 4}};

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

    int my_matrix[N], result_vector[N];
    int my_vector[N / process_count];
    MPI_Scatter(&matrix, N, MPI_INT, &my_matrix, N, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Scatter(&vector, N / process_count, MPI_INT, &my_vector, N / process_count, MPI_INT, root, MPI_COMM_WORLD);

    printf("Process %d - My sub matrix:\t", my_id);
    for (int i = 0; i < N; i++) {
        printf("%d\t", my_matrix[i]);
    }
    printf("\n");

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

    int sum = 0;
    for (int i = 0; i < process_count; i++) {
        printf("Process %d - My sub vector:\t", my_id);
        for (int i = 0; i < N / process_count; i++) {
            printf("%d\t", my_vector[i]);
        }
        printf("\n");
        for (int j = 0; j < N / process_count; j++) {
            sum += my_vector[j] * my_matrix[i * N / process_count + j];
        }
        MPI_Send(&my_vector, N / process_count, MPI_INT, prev_process, 0, MPI_COMM_WORLD);
        MPI_Recv(&my_vector, N / process_count, MPI_INT, next_process, 0, MPI_COMM_WORLD, &status);
    }

    MPI_Gather(&sum, 1, MPI_INT, &result_vector, 1, MPI_INT, root, MPI_COMM_WORLD);

    if (root == my_id) {
        printf("My result vector:\t");
        for (int i = 0; i < N; i++) {
            printf("%d\t", result_vector[i]);
        }
        printf("\n");
    }

    exit(0);
}