#include <cuda_runtime.h>
#include "reduction.cuh"

#define FLOAT4(ptr) (*reinterpret_cast<float4*>(ptr))

template <int VEC_SIZE = 4>
__global__ void LayoutNorm(float *output, float *input, int N, int K) {
    int data_idx = blockIdx.x * K + threadIdx.x * 4;

    float reg_buf[VEC_SIZE];
    FLOAT4(reg_buf) = FLOAT4(input + data_offset);

    // thread sum
    float sum = 0;
#pragma unroll
    for (int i = 1; i < VEC_SIZE; ++i) {
        sum += reg_buf[i];
    }

    // row sum & row mean
    sum = BlockReduceSum(sum);
    float mean = sum / K;

    // variance
    



}