#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "utils.cuh"
#include "reduction.cuh"
#include "dtype_utils.cuh"


#define FLOAT4(ptr) (*reinterpret_cast<float4*>(ptr))


template <int SIZE>
__inline__ __device__ float FloatVecMax(float* ptr) {
    float val = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < SIZE; ++i) {
        val = max(val, ptr[i]);
    }
}

/*
 * Args:
 *     input: M * N, 
 *     output: M * N
 * 
 * One thread block for one row.
 * grid: (M)
 * block: (N/4)
*/
template <int VEC_SIZE=4>
__global__ void SoftmaxBlock(float* output, float* input, int N) {

    const int data_offset = blockIdx.x * N + threadIdx.x * VEC_SIZE;

    float reg_buf[VEC_SIZE];
    FLOAT4(reg_buf) = FLOAT4(input + data_offset);

    // Get row max
    float thread_max = FloatVecMax<VEC_SIZE>(reg_buf);
    float row_max = BlockReduceMax(thread_max);

    // Get sum of exp
    float thread_sum_exp = 0;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float exp_x = expf(reg_buf[i] - row_max);
        thread_sum_exp += exp_x;
        reg_buf[i] = exp_x;
    }
    float row_sum_exp = BlockReduceSum(thread_sum_exp);

    // Calculate softmax
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        reg_buf[i] /= row_sum_exp;
    }

    // Save to output
    FLOAT4(output + data_offset) = FLOAT4(reg_buf);
}


void LaunchSoftmaxBlock(float* d_output, float* d_input, int M, int N) {
    dim3 grid(M);
    dim3 block(N / 4);
    SoftmaxBlock<4><<<grid, block>>>(d_output, d_input, N);
    cudaErrCheck(cudaDeviceSynchronize());
}


/*
 * cuDNN softmax wrapper for row-wise softmax
*/
void SoftmaxCUDNN(float* d_output, float* d_input, int M, int N) {
    cudnnHandle_t cudnn;
    cudnnErrCheck(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnErrCheck(cudnnCreateTensorDescriptor(&inputDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&outputDesc));

    // Set the tensor descriptor for a 2D array with row-wise softmax
    cudnnErrCheck(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, M, 1, 1, N));
    cudnnErrCheck(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, M, 1, 1, N));

    float alpha = 1.0f, beta = 0.0f;
    cudnnErrCheck(cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, inputDesc, d_input, &beta, outputDesc, d_output));

    cudnnErrCheck(cudnnDestroyTensorDescriptor(inputDesc));
    cudnnErrCheck(cudnnDestroyTensorDescriptor(outputDesc));
    cudnnErrCheck(cudnnDestroy(cudnn));
}

void CheckCorrectness(float* h_output1, float* h_output2, int size) {
    for (int i = 0; i < size; ++i) {
        // std::cout << h_output1[i] << " vs " << h_output2[i] << std::endl;
        if (std::abs(h_output1[i] - h_output2[i]) > 1e-5) {
            std::cerr << "Results differ at index " << i << ": " << h_output1[i] << " vs " << h_output2[i] << std::endl;
            return;
        }
    }
    std::cout << "Results are correct!" << std::endl;
}

int main() {
    int M = 1024; // number of rows
    int N = 4096; // number of columns

    std::vector<float> h_input(M * N);
    std::vector<float> h_output1(M * N);
    std::vector<float> h_output2(M * N);

    // Fill input with random values
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < M * N; ++i) {
        h_input[i] = dis(gen);
    }

    float *d_input, *d_output1, *d_output2;
    cudaErrCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaErrCheck(cudaMalloc(&d_output1, M * N * sizeof(float)));
    cudaErrCheck(cudaMalloc(&d_output2, M * N * sizeof(float)));

    cudaErrCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Measure time for CUDA kernel
    auto start = std::chrono::high_resolution_clock::now();
    LaunchSoftmaxBlock(d_output1, d_input, M, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cuda_duration = end - start;
    std::cout << "CUDA Kernel Time: " << cuda_duration.count() << " ms" << std::endl;

    cudaErrCheck(cudaMemcpy(h_output1.data(), d_output1, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Measure time for cuDNN
    start = std::chrono::high_resolution_clock::now();
    SoftmaxCUDNN(d_output2, d_input, M, N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cudnn_duration = end - start;
    std::cout << "cuDNN Time: " << cudnn_duration.count() << " ms" << std::endl;

    cudaErrCheck(cudaMemcpy(h_output2.data(), d_output2, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Check correctness
    CheckCorrectness(h_output1.data(), h_output2.data(), M * N);

    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);

    return 0;
}