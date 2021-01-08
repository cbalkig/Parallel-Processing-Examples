#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    MPI_Comm new_comm, inter_comm;
    int my_id, my_new_id, color, count, err;
    MPI_Status status;

    err = MPI_Init(&argc, &argv);
    if (err != 0) {
        printf("Init Error %d.", err);
        exit(-1);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &count);

    int msg = 1;
    if (count % 2 == 0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
        color = my_id % 2;
        MPI_Comm_split(MPI_COMM_WORLD, color, my_id, &new_comm);
        MPI_Comm_rank(new_comm, &my_new_id);
        printf("My ID: %d - Color: %d - My New ID: %d\n", my_id, color, my_new_id);

        if (my_id % 2 == 0) {
            MPI_Intercomm_create(new_comm, 0, MPI_COMM_WORLD, 1, 99, &inter_comm);
            printf("My ID: %d - Channel OK.\n", my_new_id);
            MPI_Send(&msg, 1, MPI_INT, my_new_id, 0, inter_comm);
            printf("Message sent from ID: %d to ID: %d\n", my_new_id, my_new_id);
        }
        else {
            MPI_Intercomm_create(new_comm, 0, MPI_COMM_WORLD, 0, 99, &inter_comm);
            printf("My ID: %d - Channel OK.\n", my_new_id);
            MPI_Recv(&msg, 1, MPI_INT, my_new_id, 0, inter_comm, &status);
            printf("Message received from ID: %d to ID: %d\n", my_new_id, my_new_id);
            printf("%d - %d - Msg arrived: %d\n", my_new_id, my_new_id, msg);
        }
    }

    MPI_Comm_free(&inter_comm);
    MPI_Comm_free(&new_comm);

    return 0;
}