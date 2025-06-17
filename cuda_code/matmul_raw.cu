#include <iostream>
#include "cuda_runtime.h"
using namespace std;
__global__ void matmul(float *cuda_A, float *cuda_B, float *cuda_C, int n, int k, int m)
{
    // C[n][m]
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < m && y < n)
    {
        float sum = 0.0;
        for (int i = 0; i < k; ++i)
        {
            sum += cuda_A[y * k + i] * cuda_B[x + m * i];
        }
        cuda_C[y * m + x] = sum;
    }
}
int main()
{
    int n = 256;
    int k = 256;
    int m = 256;
    // A[n][k] * B[k][m] = C[n][m];
    float *A, *B, *C;
    float *cuda_A, *cuda_B, *cuda_C;
    A = new float[n * k];
    B = new float[k * m];
    C = new float[n * m];
    cudaMalloc((void **)&cuda_A, n * k * sizeof(float));
    cudaMalloc((void **)&cuda_B, m * k * sizeof(float));
    cudaMalloc((void **)&cuda_C, m * n * sizeof(float));
    for (int i = 0; i < n * k; ++i)
    {
        A[i] = 2.0;
    }
    for (int i = 0; i < m * k; ++i)
    {
        B[i] = 3.0;
    }
    cudaMemcpy(cuda_A, A, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B, m * k * sizeof(float), cudaMemcpyHostToDevice);
    dim3 grid_dim(10, 10);
    dim3 block_dim(32, 32);
    matmul<<<grid_dim, block_dim>>>(cuda_A, cuda_B, cuda_C, n, k, m);
    cudaMemcpy(C, cuda_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            cout << C[i * m + j] << ' ';
        }
        cout << endl;
    }
    delete A, B, C;
    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);

    return 0;
}