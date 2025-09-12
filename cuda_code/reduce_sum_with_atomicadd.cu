#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

__global__ void reduce_sum(float *d_A, float *out, int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < n)
    {
        sdata[tid] = d_A[i];
    }
    else
    {
        sdata[tid] = 0.0f; // Handle out-of-bounds
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0)
    {
        atomicAdd(out, sdata[0]);
    }
}

int main()
{
    const int N = 1024;
    float *h_A = new float[N];
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i + 1); // Initialize with some values
    }
    float *d_A, *d_out;
    float h_out = 0.0f;
    size_t size = N * sizeof(float);
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_out, sizeof(float));
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, &h_out, sizeof(float), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    reduce_sum<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_A, d_out, N);
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    float total;
    for (int i = 0; i < N; ++i)
    {
        total += h_A[i];
    }
    cout << "Expected Sum: " << total << endl;
    if (cudaGetLastError() != cudaSuccess)
    {
        cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << endl;
        return -1;
    }
    cout << "Sum: " << h_out << endl;

    cudaFree(d_A);
    cudaFree(d_out);
    delete[] h_A;
    return 0;
}