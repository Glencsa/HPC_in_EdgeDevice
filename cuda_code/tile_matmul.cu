#include <iostream>
#include "cuda_runtime.h"
using namespace std;
__global__ void matmul(float *d_A, float *d_B, float *d_C, int n, int m, int k)
{
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    int b_x = blockIdx.x;
    int b_y = blockIdx.y;
    __shared__ float s_A[32 * 32];
    __shared__ float s_B[32 * 32];
    d_A += 32 * k * b_y;
    d_B += 32 * b_x;
    float tmp[32] = {0.0};
    for (int i = 0; i < k; i += 32)
    {
        s_A[t_y * 32 + t_x] = d_A[t_y * 32 + t_x + i * 32];
        s_B[t_y * 32 + t_x] = d_B[t_y * 32 + t_x + i * 32];
        __syncthreads();
        for (int j = 0; j < 32; ++j)
        {
            for (int l = 0; l < 32; ++l)
            {
                tmp[j] += s_A[t_y * 32 + l] * s_B[j + l * 32];
            }
        }
        __syncthreads();
        d_A += 32;
        d_B += m * 32;
    }
    for (int i = 0; i < 32; ++i)
    {
        d_C[b_y * 32 * m + i] = tmp[i];
    }
}
int main()
{
    int n = 256;
    int m = 512;
    int k = 128;
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    A = new float[n * k];
    B = new float[m * k];
    C = new float[n * m];
    cudaMalloc((void **)d_A, sizeof(float) * n * k);
    cudaMalloc((void **)d_B, sizeof(float) * m * k);
    cudaMalloc((void **)d_C, sizeof(float) * m * n);
    for (int i = 0; i < n * k; ++i)
    {
        A[i] = 1;
    }
    for (int i = 0; i < m * k; ++i)
    {
        B[i] = 1;
    }
    cudaMemcpy(d_A, A, sizeof(float) * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    dim3 block_dim(32, 32);
    dim3 grid_dim(256 / 32, 512 / 32);
    matmul<<<grid_dim, block_dim>>>(d_A, d_B, d_C, n, m, k);
    return 0;
}