#include "kernels/sgemm_v1_thread_block_tile.cuh"
#include "kernels/sgemm_v2_optimized.cuh"
#include "kernels/sgemm_v3_optimized_A100.cuh"
#include "runner.cuh"
#include "utils.cuh"
#include <iostream>

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

void RunCublasFp32(cublasHandle_t handle, int M, int N, int K, float alpha, float *d_A, float *d_B,
                   float beta, float *d_C)
{
    // cuBLAS uses column-major order. So we need to transpose the matrix A and B.
    cublasErrCheck(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_B, K, d_A, N,
                               &beta, d_C, N));
}

void RunSgemmV1ThreadBlockTile(int M, int N, int K, float alpha, float *d_A, float *d_B, float beta,
                               float *d_C)
{
    constexpr int BLOCK_TILE_SIZE = 16;
    dim3 gridDim(CEIL_DIV(N, BLOCK_TILE_SIZE), CEIL_DIV(M, BLOCK_TILE_SIZE), 1);
    dim3 blockDim(BLOCK_TILE_SIZE, BLOCK_TILE_SIZE, 1);
    SgemmV1<BLOCK_TILE_SIZE><<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
}

void RunSgemmV2Optimized(int M, int N, int K, float alpha, float *d_A, float *d_B, float beta,
                         float *d_C)
{
    constexpr int BLOCK_ITEMS_M = 128;
    constexpr int BLOCK_ITEMS_N = 128;
    constexpr int BLOCK_ITEMS_K = 8;
    constexpr int THREAD_ITEMS_M = 8;
    constexpr int THREAD_ITEMS_N = 8;
    int total_thread_num = (BLOCK_ITEMS_M / THREAD_ITEMS_M) * (BLOCK_ITEMS_N / THREAD_ITEMS_N);

    dim3 gridDim(CEIL_DIV(N, BLOCK_ITEMS_N), CEIL_DIV(M, BLOCK_ITEMS_M), 1);
    dim3 blockDim(total_thread_num, 1, 1);
    SgemmV2<BLOCK_ITEMS_M, BLOCK_ITEMS_N, BLOCK_ITEMS_K, THREAD_ITEMS_M, THREAD_ITEMS_N>
        <<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
}

void RunSgemmV3OptimizedA100(int M, int N, int K, float alpha, float *d_A, float *d_B, float beta,
                             float *d_C)
{
    constexpr int BLOCK_ITEMS_M = 128;
    constexpr int BLOCK_ITEMS_N = 128;
    constexpr int BLOCK_ITEMS_K = 16;
    constexpr int THREAD_ITEMS_M = 8;
    constexpr int THREAD_ITEMS_N = 8;
    int total_thread_num = (BLOCK_ITEMS_M / THREAD_ITEMS_M) * (BLOCK_ITEMS_N / THREAD_ITEMS_N);

    dim3 gridDim(CEIL_DIV(N, BLOCK_ITEMS_N), CEIL_DIV(M, BLOCK_ITEMS_M), 1);
    dim3 blockDim(total_thread_num, 1, 1);
    SgemmV3<BLOCK_ITEMS_M, BLOCK_ITEMS_N, BLOCK_ITEMS_K, THREAD_ITEMS_M, THREAD_ITEMS_N>
        <<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
}

void RunSgemmKernel(int kernel_num, int M, int N, int K, float alpha, float *d_A, float *d_B,
                    float beta, float *d_C, cublasHandle_t handle)
{
    switch (kernel_num)
    {
    case 0:
        RunCublasFp32(handle, M, N, K, alpha, d_A, d_B, beta, d_C);
        break;
    case 1:
        RunSgemmV1ThreadBlockTile(M, N, K, alpha, d_A, d_B, beta, d_C);
        break;
    case 2:
        RunSgemmV2Optimized(M, N, K, alpha, d_A, d_B, beta, d_C);
        break;
    case 3:
        RunSgemmV3OptimizedA100(M, N, K, alpha, d_A, d_B, beta, d_C);
        break;
    default:
        std::cerr << "Invalid kernel number: " << kernel_num << std::endl;
        break;
    }
}