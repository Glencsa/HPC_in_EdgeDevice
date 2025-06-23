#include <iostream>
#include "cuda_runtime.h"
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
using namespace std;
// 转变为每个cuda核心计算C矩阵当中的某一“块”，而不是某一个元素
void randomize_matrix(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = 1;
    }
}
__global__ void shared_matmul(float *A, float *B, float *C, int n, int m, int k)
{
    __shared__ float shared_A[32 * 32];
    __shared__ float shared_B[32 * 32];
    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;
    const uint thread_row = threadIdx.y;
    const uint thread_col = threadIdx.x;
    A += c_row * 32 * k;
    B += c_col * 32;
    C += c_row * 32 * m + c_col * 32;
    float tmp = 0.0;
    for (int bl = 0; bl < k; bl += 32)
    {
        shared_A[thread_row * 32 + thread_col] = A[thread_row * 32 + thread_col];
        shared_B[thread_row * 32 + thread_col] = B[thread_row * 32 + thread_col];
        __syncthreads();

        for (int j = 0; j < 32; ++j)
        {
            tmp += shared_A[thread_row * 32 + j] * shared_B[thread_col + 32 * j];
        }
        __syncthreads();
        A += 32;
        B += 32 * m;
    }
    C[thread_row * m + thread_col] = tmp;
}
int main()
{
    int n = 256;
    int m = 256;
    int k = 128;
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    float *C_ref;
    A = new float[n * k];
    B = new float[m * k];
    C = new float[n * m];
    C_ref = new float[n * m];
    cudaMalloc((void **)&d_A, n * k * sizeof(float));
    cudaMalloc((void **)&d_B, m * k * sizeof(float));
    cudaMalloc((void **)&d_C, m * n * sizeof(float));

    // Initialize matrices
    randomize_matrix(A, n * k);
    randomize_matrix(B, k * m);

    cudaMemcpy(d_A, A, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, m * k * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block_dim(32, 32); // 二维
    dim3 grid_dim(CEIL_DIV(n, 32), CEIL_DIV(m, 32));
    shared_matmul<<<grid_dim, block_dim>>>(d_A, d_B, d_C, n, m, k);
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i)
    {
        cout << A[i] << ' ' << B[i] << ' ';
        cout << C[i] << ' ';
        if (i % 10 == 0)
            cout << endl;
    }
    delete A, B, C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
