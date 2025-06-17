#include <iostream>
#include "cuda_runtime.h"
using namespace std;
__global__ void mul(float *d_A, float *d_B, float *d_C, int n)
{
    uint x = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x;
    if (x < 100)
    {
        d_C[x] = d_A[x] + d_B[x];
    }
}
int main()
{
    dim3 block_dim(32, 32);
    dim3 grid_dim(32, 32);
    int n = 100;
    float *A, *B, *C;
    float *cuda_A, *cuda_B, *cuda_C;

    cudaMalloc((void **)&cuda_A, n * sizeof(float));
    cudaMalloc((void **)&cuda_B, n * sizeof(float));
    cudaMalloc((void **)&cuda_C, n * sizeof(float));
    A = new float[100];
    B = new float[100];
    C = new float[100];
    for (int i = 0; i < 100; ++i)
    {
        A[i] = i;
        B[i] = i + 1;
    }
    cudaMemcpy(cuda_A, A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B, n * sizeof(float), cudaMemcpyHostToDevice);
    mul<<<grid_dim, block_dim>>>(cuda_A, cuda_B, cuda_C, n);
    cudaMemcpy(C, cuda_C, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 100; ++i)
    {
        cout << C[i] << " ";
        if (i % 9 == 0)
            cout << endl;
    }
    free(A);
    free(B);
    free(C);
    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);

    return 0;
}