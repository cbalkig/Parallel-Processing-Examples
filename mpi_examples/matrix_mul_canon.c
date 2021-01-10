#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

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

    int slide = (N / sqrt(process_count));
    int my_matrixA[slide][slide], my_matrixB[slide][slide];
    int matrixC[N][N];

    MPI_Datatype matrixAB_data_type;
    MPI_Type_vector(slide, slide, N, MPI_INT, &matrixAB_data_type);
    MPI_Type_commit(&matrixAB_data_type);

    MPI_Datatype matrixC_data_type;
    MPI_Type_vector(slide, slide, N, MPI_INT, &matrixC_data_type);
    MPI_Type_commit(&matrixC_data_type);

    if (root == my_id) {
        for (int p = 1; p < process_count; p++) {
            int col = (p / slide) * slide;
            int row = (p % slide) * slide;
            MPI_Send(&matrixA[col][row], 1, matrixAB_data_type, p, 0, MPI_COMM_WORLD);
            MPI_Send(&matrixB[col][row], 1, matrixAB_data_type, p, 0, MPI_COMM_WORLD);
        }
        MPI_Sendrecv(&matrixA[0][0], 1, matrixAB_data_type, root, 0, &my_matrixA, N * N / process_count, MPI_INT, root,
                     0, MPI_COMM_WORLD,
                     &status);
        MPI_Sendrecv(&matrixB[0][0], 1, matrixAB_data_type, root, 0, &my_matrixB, N * N / process_count, MPI_INT, root,
                     0,
                     MPI_COMM_WORLD, &status);
    } else {
        MPI_Recv(&my_matrixA, N * N / process_count, MPI_INT, root, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&my_matrixB, N * N / process_count, MPI_INT, root, 0, MPI_COMM_WORLD, &status);
    }

    int sum[slide][slide];
    for (int i = 0; i < slide; i++) {
        for (int j = 0; j < slide; j++) {
            sum[i][j] = 0;
        }
    }

    for (int iter = 0; iter < N; iter++) {
        printf("Iteration %d - Process %d - My sub matrix A:\t", iter, my_id);
        for (int i = 0; i < slide; i++) {
            for (int j = 0; j < slide; j++) {
                printf("%d\t", my_matrixA[i][j]);
            }
        }
        printf("\n");

        printf("Iteration %d - Process %d - My sub matrix B:\t", iter, my_id);
        for (int i = 0; i < slide; i++) {
            for (int j = 0; j < slide; j++) {
                printf("%d\t", my_matrixB[i][j]);
            }
        }
        printf("\n");


        for (int i = 0; i < slide; i++) {
            for (int j = 0; j < slide; j++) {
                sum[i][j] += my_matrixA[i][j] * my_matrixB[i][j];
            }
        }

        printf("Iteration %d - Process %d - My sum matrix:\t", iter, my_id);
        for (int i = 0; i < slide; i++) {
            for (int j = 0; j < slide; j++) {
                printf("%d\t", sum[i][j]);
            }
        }
        printf("\n");

        int willSend[slide];
        for (int i = 0; i < slide; i++) {
            for (int j = slide - 2; j >= 0; j--) {
                if (j == 0) {
                    willSend[i] = my_matrixA[i][j];
                }
                my_matrixA[i][j] = my_matrixA[i][j + 1];
            }
        }

        int peer_process_A;
        int mod = sqrt(process_count);
        if (my_id % mod == 0) {
            peer_process_A = my_id + 1;
            for (int i = 0; i < slide; i++) {
                MPI_Send(&my_matrixA[i][0], 1, MPI_INT, peer_process_A, 0, MPI_COMM_WORLD);
                MPI_Recv(&my_matrixA[i][slide - 1], 1, MPI_INT, peer_process_A, 0, MPI_COMM_WORLD, &status);
            }
        } else {
            peer_process_A = my_id - 1;
            for (int i = 0; i < slide; i++) {
                MPI_Recv(&my_matrixA[i][slide - 1], 1, MPI_INT, peer_process_A, 0, MPI_COMM_WORLD, &status);
                MPI_Send(&my_matrixA[i][0], 1, MPI_INT, peer_process_A, 0, MPI_COMM_WORLD);
            }
        }

        willSend[slide];
        for (int i = 0; i < slide; i++) {
            for (int j = slide - 2; j >= 0; j--) {
                if (j == 0) {
                    willSend[i] = my_matrixB[j][i];
                }
                my_matrixB[j][i] = my_matrixB[j + 1][i];
            }
        }

        int peer_process_B;
        if (my_id / mod == 0) {
            peer_process_B = my_id + mod;
            for (int i = 0; i < slide; i++) {
                MPI_Send(&willSend[i], 1, MPI_INT, peer_process_B, 0, MPI_COMM_WORLD);
                MPI_Recv(&my_matrixB[slide - 1][i], 1, MPI_INT, peer_process_B, 0, MPI_COMM_WORLD, &status);
            }
        } else {
            peer_process_B = my_id - mod;
            for (int i = 0; i < slide; i++) {
                MPI_Recv(&my_matrixB[slide - 1][i], 1, MPI_INT, peer_process_B, 0, MPI_COMM_WORLD, &status);
                MPI_Send(&willSend[i], 1, MPI_INT, peer_process_B, 0, MPI_COMM_WORLD);
            }
        }
    }

    if (root == my_id) {
        MPI_Sendrecv(&sum, slide * slide, MPI_INT, root, 0, &matrixC, 1, matrixC_data_type, root, 0, MPI_COMM_WORLD,
                     &status);
        for (int p = 1; p < process_count; p++) {
            int col = (p / slide) * slide;
            int row = (p % slide) * slide;
            MPI_Recv(&matrixC[col][row], 1, matrixC_data_type, p, 0, MPI_COMM_WORLD, &status);
        }
    } else {
        MPI_Send(&sum, slide * slide, MPI_INT, root, 0, MPI_COMM_WORLD);
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