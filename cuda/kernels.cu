#include <cuda.h>

__global__
void mykernel () {
    printf("Hello from %d\n", threadIdx.x);
}

__global__ void VecAdd(float* A, float* B, float* C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; int j = blockIdx.y * blockDim.y + threadIdx.y; if (i < N && j < N)
}

int main() {
    float A[N], B[N], C[N];

    nblocks = N / 512;
    mykernel <<< nblocks, 16 >>> ();
    VecAdd<<<1, N>>>(A, B, C);

    N = 16
    int numBlocks = 1; // N = 16 --> 16 * 16 < 512, 1 block yeter.
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C)

    N = 1024
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y); // N = 1024 --> 1024 * 1024 > 512, 1 block yetmez.
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C)

    cudaDeviceSynchronize();
    return 0;
}