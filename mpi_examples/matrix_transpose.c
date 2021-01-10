#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#define N 4

int main(int argc, char *argv[]) {
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

    int slide = (N / sqrt(process_count));
    int my_matrix[slide][slide];
    int final_matrix[N][N];

    MPI_Datatype matrix_data_type;
    MPI_Type_vector(slide, slide, N, MPI_INT, &matrix_data_type);
    MPI_Type_commit(&matrix_data_type);

    MPI_Datatype matrix_final_data_type;
    MPI_Type_vector(slide, slide, N, MPI_INT, &matrix_final_data_type);
    MPI_Type_commit(&matrix_final_data_type);

    if (root == my_id) {
        for (int p = 1; p < process_count; p++) {
            int col = (p / slide) * slide;
            int row = (p % slide) * slide;
            MPI_Send(&matrix[col][row], 1, matrix_data_type, p, 0, MPI_COMM_WORLD);
        }
        MPI_Sendrecv(&matrix[0][0], 1, matrix_data_type, root, 0, &my_matrix, N * N / process_count, MPI_INT, root,
                     0, MPI_COMM_WORLD,
                     &status);
    } else {
        MPI_Recv(&my_matrix, N * N / process_count, MPI_INT, root, 0, MPI_COMM_WORLD, &status);
    }

    for (int iter = 0; iter < sqrt(N) - 1; iter++) {
        printf("Iteration %d - Process %d - My sub matrix:\t", iter, my_id);
        for (int i = 0; i < slide; i++) {
            for (int j = 0; j < slide; j++) {
                printf("%d\t", my_matrix[i][j]);
            }
        }
        printf("\n");

        int subSlider = pow(2, iter);
        int window = pow(2, iter + 1);
        for (int i = 0; i < slide; i = i + window) {
            for (int j = subSlider; j < slide; j = j + window) {
                for (int k = 0; k < subSlider; k++) {
                    int temp = my_matrix[i][j + k];
                    my_matrix[i][j + k] = my_matrix[j + k][i];
                    my_matrix[j + k][i] = temp;
                }
            }
        }

        if (root == my_id) {
            MPI_Sendrecv(&my_matrix, slide * slide, MPI_INT, root, 0, &final_matrix, 1, matrix_final_data_type, root, 0,
                         MPI_COMM_WORLD,
                         &status);
            for (int p = 1; p < process_count; p++) {
                int col = (p / slide) * slide;
                int row = (p % slide) * slide;
                MPI_Recv(&final_matrix[col][row], 1, matrix_final_data_type, p, 0, MPI_COMM_WORLD, &status);
            }

            printf("Iteration %d - My result matrix:\t", iter);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    printf("%d\t", final_matrix[i][j]);
                }
                printf("\n");
            }
            printf("\n");
        } else {
            MPI_Send(&my_matrix, slide * slide, MPI_INT, root, 0, MPI_COMM_WORLD);
        }
    }

    if (my_id == 1) {
        MPI_Send(&my_matrix, slide * slide, MPI_INT, root, 0, MPI_COMM_WORLD);
    }

    if (my_id == 2) {
        MPI_Send(&my_matrix, slide * slide, MPI_INT, root, 0, MPI_COMM_WORLD);
    }

    if (root == my_id) {
        MPI_Recv(&final_matrix[slide][0], 1, matrix_final_data_type, 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&final_matrix[0][slide], 1, matrix_final_data_type, 2, 0, MPI_COMM_WORLD, &status);

        printf("My result matrix:\t");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d\t", final_matrix[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    } else {
        MPI_Send(&my_matrix, slide * slide, MPI_INT, root, 0, MPI_COMM_WORLD);
    }

    exit(0);
}