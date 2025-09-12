#include <iostream>
#include "cuda_runtime.h"
using namespace std;
const int len = 1024;
__global__ void reduce_add_kernel(int *d_arr, int *d_out, int len)
{
    __shared__ int s_data[32];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len)
    {
        s_data[threadIdx.x] = d_arr[i];
    }
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (threadIdx.x % (2 * s) == 0 && i + s < len)
        {
            s_data[threadIdx.x] += s_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        d_out[blockIdx.x] = s_data[0];
    }
}
int main()
{
    int *arr = new int[len];
    int *out = new int[len];
    int *d_arr, *d_out;
    for (int i = 0; i < len; ++i)
    {
        arr[i] = i;
    }
    cudaMalloc((void **)&d_arr, sizeof(int) * len);
    cudaMalloc((void **)&d_out, sizeof(int) * len);

    cudaMemcpy(d_arr, arr, sizeof(int) * len, cudaMemcpyHostToDevice);
    dim3 block_dim(32);
    dim3 grid_dim(len / 32 + 2);
    reduce_add_kernel<<<block_dim, grid_dim>>>(d_arr, d_out, len);
    cudaMemcpy(out, d_out, sizeof(int) * len, cudaMemcpyDeviceToHost);
    int sum = 0;
    for (int i = 0; i < grid_dim.x; ++i)
    {
        cout << out[i] << ' ';
        sum += out[i];
    }
    cout << endl;
    // 核对结果
    // grid_dim(len / 32 + 1) 528 1584 2640 3696 4752 5808 6864 7920 8976 10032 11088 12144 13200 14256 15312 16368 17424 18480 19536 20592 21648 22704 23760 24816 25872 26928 27984 29040 30096 31152 32208 32240 0
    // grid_dim(len / 32) 496 1520 2544 3568 4592 5616 6640 7664 8688 9712 10736 11760 12784 13808 14832 15856 16880 17904 18928 19952 20976 22000 23024 24048 25072 26096 27120 28144 29168 30192 31216 32240
    int sum2 = 0;
    for (int i = 0; i < len; i++)
    {
        sum2 += arr[i];
    }
    cout << "cuda result :" << sum << endl;
    cout << "cup result :" << sum2 << endl;
    return 0;
}