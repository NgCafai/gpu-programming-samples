#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <cmath>

// define some error checking macros
#define cudaErrCheck(stat) \
    { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

#define cublasErrCheck(stat) \
    { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

#define curandErrCheck(stat) \
    { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
    if (stat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
    }
}

// Check if two matrices are equal
bool IsMatrixEqual(float *A, float *B, int M, int N, float tol = 1e-5f) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(A[i] - B[i]) > tol) {
            return false;
        }
    }
    return true;
}