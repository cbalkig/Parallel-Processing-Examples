#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define N 4

int main(int argc, char *argv[]) {
    int matrixA[N][N] = {{1, 1, 1, 1},
                         {2, 2, 2, 2},
                         {3, 3, 3, 3},
                         {4, 4, 4, 4}};
    int matrixB[N][N] = {{1, 1, 1, 1},
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

    int my_matrixA[N * N / process_count], my_matrixB[N * N / process_count];
    int matrixC[N][N];

    MPI_Datatype matrixB_data_type;
    MPI_Type_vector(N, 1, N, MPI_INT, &matrixB_data_type);
    MPI_Type_commit(&matrixB_data_type);

    for (int iter = 0; iter < process_count; iter++) {
        if (root == my_id) {
            for (int p = 1; p < process_count; p++) {
                MPI_Send(&matrixA[iter], N, MPI_INT, p, 0, MPI_COMM_WORLD);
                MPI_Send(&matrixB[0][p], 1, matrixB_data_type, p, 0, MPI_COMM_WORLD);
            }
            MPI_Sendrecv(&matrixA[iter], N, MPI_INT, root, 0, &my_matrixA, N, MPI_INT, root, 0, MPI_COMM_WORLD,
                         &status);
            MPI_Sendrecv(&matrixB[0][0], 1, matrixB_data_type, root, 0, &my_matrixB, N, MPI_INT, root, 0,
                         MPI_COMM_WORLD, &status);
        } else {
            MPI_Recv(&my_matrixA, N, MPI_INT, root, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&my_matrixB, N, MPI_INT, root, 0, MPI_COMM_WORLD, &status);
        }

        printf("Process %d - My sub matrix A:\t", my_id);
        for (int i = 0; i < N; i++) {
            printf("%d\t", my_matrixA[i]);
        }
        printf("\n");

        printf("Process %d - My sub matrix B:\t", my_id);
        for (int i = 0; i < N; i++) {
            printf("%d\t", my_matrixB[i]);
        }
        printf("\n");

        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += my_matrixA[i] * my_matrixB[i];
        }

        MPI_Gather(&sum, 1, MPI_INT, &matrixC[iter], 1, MPI_INT, root, MPI_COMM_WORLD);
    }

    if (root == my_id) {
        printf("My result matrix:\t");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d\t", matrixC[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    exit(0);
}