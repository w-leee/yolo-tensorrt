#include <cuda_runtime.h>

__global__ void hwc_to_chw_kernel(const unsigned char* src, float* dst, int total_pixels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_pixels)
    {
        int offset = i * 3;
        unsigned char r = src[offset];
        unsigned char g = src[offset + 1];
        unsigned char b = src[offset + 2];

        dst[0 * total_pixels + i] = r / 255.0f;
        dst[1 * total_pixels + i] = g / 255.0f;
        dst[2 * total_pixels + i] = b / 255.0f;
    }
}

void launch_preprocess_kernel(const unsigned char* d_src, float* d_dst, int total_pixels)
{
    int threads_per_block = 256;
    int blocks = (total_pixels + threads_per_block - 1) / threads_per_block;
    hwc_to_chw_kernel<<<blocks, threads_per_block>>>(d_src, d_dst, total_pixels);
}