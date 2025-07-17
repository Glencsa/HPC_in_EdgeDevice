#include <iostream>
#include "cuda_runtime.h"
using namespace std;
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
// blockdim = (32,32)
__global__ void matmul(float *A, float *B, float *C, int N, int K, int M)
{
    __shared__ float A_s[32 * 32];
    __shared__ float B_s[32 * 32];
    int xtid = threadIdx.x;
    int ytid = threadIdx.y;
    int blocksize = 32;
    A += K * blockIdx.y * blocksize;
    B += blockIdx.x * blocksize;
    C += blockIdx.y * blocksize * M + blockIdx.x * blocksize;
    float tmp = 0.0;
    for (int i = 0; i < K; i += 32)
    {
        A_s[ytid * blocksize + xtid] = A[ytid * K + i + xtid];
        B_s[ytid * blocksize + xtid] = B[(ytid + i) * M + xtid];
        __syncthreads();
        for (int j = 0; j < 32; ++j)
        {
            tmp += A_s[ytid * blocksize + j] * B_s[j * blocksize + xtid];
        }
        A += blocksize;
        B += blocksize * M;
    }
    C[ytid * M + xtid] = tmp;
}

int main()
{
    float *A, *B, *C, *d_A, *d_B, *d_C;
    int N = 256, K = 128, M = 512;
    A = new float[256 * 128];
    B = new float[128 * 512];
    C = new float[256 * 512];
    cudaMalloc((void **)&d_A, sizeof(float) * N * K);
    cudaMalloc((void **)&d_B, sizeof(float) * K * M);
    cudaMalloc((void **)&d_C, sizeof(float) * N * M);
    for (int i = 0; i < N * K; ++i)
    {
        A[i] = 1;
    }
    for (int i = 0; i < M * K; ++i)
    {
        B[i] = 1;
    }
    cudaMemcpy(d_A, A, sizeof(float) * N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    matmul<<<grid, block>>>(d_A, d_B, d_C, N, K, M);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N * M; ++i)
    {
        cout << C[i] << ' ';
        if (i % 30 == 0)
            cout << endl;
    }
    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}