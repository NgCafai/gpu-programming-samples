#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

void RunSgemmKernel(int kernel_num, int M, int N, int K, float alpha, float *d_A, float *d_B,
                    float beta, float *d_C, cublasHandle_t handle);