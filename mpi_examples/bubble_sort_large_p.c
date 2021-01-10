#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#define N 8

int main(int argc, char *argv[]) {
    int numbers[N] = {10, 8, 2, 5, 1, 6, 9, 3};

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

    int my_number;
    int peer_number;
    MPI_Scatter(&numbers, 1, MPI_INT, &my_number, 1, MPI_INT, root, MPI_COMM_WORLD);

    for (int i = 0; i < N - 1; i++) {
        if (i % 2 == 0) {
            if (my_id % 2 == 0) {
                MPI_Send(&my_number, 1, MPI_INT, my_id + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&peer_number, 1, MPI_INT, my_id + 1, 0, MPI_COMM_WORLD, &status);
                if (peer_number < my_number) {
                    my_number = peer_number;
                }
            } else {
                MPI_Recv(&peer_number, 1, MPI_INT, my_id - 1, 0, MPI_COMM_WORLD, &status);
                MPI_Send(&my_number, 1, MPI_INT, my_id - 1, 0, MPI_COMM_WORLD);
                if (peer_number > my_number) {
                    my_number = peer_number;
                }
            }
        } else {
            if (my_id % 2 != 0) {
                if (my_id + 1 >= process_count) {
                    continue;
                }
                MPI_Send(&my_number, 1, MPI_INT, my_id + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&peer_number, 1, MPI_INT, my_id + 1, 0, MPI_COMM_WORLD, &status);
                if (peer_number < my_number) {
                    my_number = peer_number;
                }
            } else {
                if (my_id - 1 < 0) {
                    continue;
                }
                MPI_Recv(&peer_number, 1, MPI_INT, my_id - 1, 0, MPI_COMM_WORLD, &status);
                MPI_Send(&my_number, 1, MPI_INT, my_id - 1, 0, MPI_COMM_WORLD);
                if (peer_number > my_number) {
                    my_number = peer_number;
                }
            }
        }
    }

    MPI_Gather(&my_number, 1, MPI_INT, &numbers, 1, MPI_INT, root, MPI_COMM_WORLD);

    if (my_id == root) {
        printf("Numbers: \t");
        for (int i = 0; i < N; i++) {
            printf("%d\t", numbers[i]);
        }
        printf("\n");
    }

    exit(0);
}