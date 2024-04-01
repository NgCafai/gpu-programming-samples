#include <cuda.h>
#include "kernel.h"

__global__ void addKernel(int *c, const int *a, const int *b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

void runAddKernel(int *c, const int *a, const int *b, int size) {
    int *dev_a, *dev_b, *dev_c;

    // Allocate GPU buffers
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));

    // Copy input arrays to the GPU
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b, size);

    // Copy the results back to the host
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
