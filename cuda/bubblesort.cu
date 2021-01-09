#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include "win-gettimeofday.h"

/* Number of threads per block */
#define THREADS_PER_BLOCK 512

__global__ void bubbleSort(int array[], int End) { //Used to sort the given array as a BubbleSort
    int swapped = 0;
    int temp;
    do {
        swapped = 0;
        for (int i = 0; i < End; i++) {
            if (array[i] > array[i + 1]) {
                temp = array[i];
                array[i] = array[i + 1];
                array[i + 1] = temp;
                swapped = 1;
            }
        }

    } while (swapped == 1);

}

void populateRandomArray(int *x, int num_elements) { //Used to populate the given array with random integers
    for (int i = 0; i < num_elements; i++) {
        x[i] = rand() % 100 + 1;
    }
}

void bubbleSortCPU(int array[], int End) { //Used to sort the given array as a BubbleSort
    int swapped = 0;
    int temp;

    do {
        swapped = 0;
        for (int i = 0; i < End; i++) {
            if (array[i] > array[i + 1]) {
                temp = array[i];
                array[i] = array[i + 1];
                array[i + 1] = temp;
                swapped = 1;
            }
        }

    } while (swapped == 1);

}

int main(void) {
    const int number_of_elements = 100000; //Total amount of elements to be sorted

    int trials[number_of_elements]; //To be used if testing requires multiple attempts

    int *host_a;//used to store the whole 1d matrix on the host
    int *host_c;//used to store the sorted 1d matrix from the device on the host

    int *device_a;//used to store the whole 1d matrix on the device
    int *device_c;//used to store the sorted 1d matrix on the device

    double cpu_time_without_allocation;
    double cpu_time_with_allocation;
    double cpu_end_time;

    double gpu_time_without_transfer;
    double gpu_time_with_transfer;
    double gpu_end_time_without_transfer;
    double gpu_end_time_with_transfer;


    //----USED FOR SERIAL IMPLEMENTATION
    //int* arrayCPU;

    //cpu_time_with_allocation = get_current_time();

    //arrayCPU = (int *)malloc(number_of_elements *sizeof(int));

    //populateRandomArray(arrayCPU, number_of_elements);

    //cpu_time_without_allocation = get_current_time();

    //bubbleSortCPU(arrayCPU, number_of_elements);

    //cpu_end_time = get_current_time();

    //printf("Number of elements = %d, CPU Time (Not including data allocation): %lfs\n", number_of_elements, (cpu_end_time - cpu_time_without_allocation));
    //printf("Number of elements = %d, CPU Time (Including data allocation): %lfs\n", number_of_elements, (cpu_end_time - cpu_time_with_allocation));

    //free(arrayCPU);
    //--------------------------------------------------------

    for (int i = 0; i < 1; i++) {
        int size = trials[i] *
                   sizeof(int); //Used to find the BYTE size, used to for memory allocation on the device and host

        int end = number_of_elements;//Used to find the end of the Matrix

        host_a = (int *) malloc(size);// Allocates the memory on the host equivilant to the size of the matrix
        host_c = (int *) malloc(size);// Allocates the memory on the host equivilant to the size of the matrix

        cudaMalloc((void **) &device_a, size);// Allocates the memory on the device equivilant to the size of the matrix
        cudaMalloc((void **) &device_c, size);// Allocates the memory on the device equivilant to the size of the matrix

        populateRandomArray(host_a, number_of_elements);//Populates the natrux with the specified amount of elements

        gpu_time_with_transfer = get_current_time(); //Gets time before the Memory allocation

        cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);

        gpu_time_without_transfer = get_current_time(); //Gets time after the memory allocation

        dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);
        dim3 dimGrid((trials[i] + dimBlock.x - 1) / dimBlock.x, 1, 1);

        bubbleSort << < dimGrid, dimBlock >> > (device_a, end); //Invokes the Kernel

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            /* Returns the description string for an error code */
            printf("Error: %s\n", cudaGetErrorString(error));
        }

        cudaThreadSynchronize(); //Waits untill all of the threads have finished

        gpu_end_time_without_transfer = get_current_time(); //Gets time without the transfer delay

        cudaMemcpy(host_c, device_a, size, cudaMemcpyDeviceToHost);

        gpu_end_time_with_transfer = get_current_time();  //Gets time with the transfer delay

        printf("Number of elements = %d, GPU Time (Not including data transfer): %lfs\n", number_of_elements,
               (gpu_end_time_without_transfer - gpu_time_without_transfer));
        printf("Number of elements = %d, GPU Time (Including data transfer): %lfs\n", number_of_elements,
               (gpu_end_time_with_transfer - gpu_time_with_transfer));

        free(host_a);//Frees the memory used on the host
        free(host_c);//Frees the memory used on the host

        cudaFree(device_a);//Frees the memory used on the device
        cudaFree(device_c);//Frees the memory used on the device

        cudaDeviceReset();
    }
    return 0;
}