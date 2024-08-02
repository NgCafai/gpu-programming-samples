#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <cmath>

// define some error checking macros
#define cudaErrCheck(stat) \
    { cudaErrCheck_((stat), __FILE__, __LINE__); }
inline void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

#define cublasErrCheck(stat) \
    { cublasErrCheck_((stat), __FILE__, __LINE__); }
inline void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

#define curandErrCheck(stat) \
    { curandErrCheck_((stat), __FILE__, __LINE__); }
inline void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
    if (stat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
    }
}

#define cudnnErrCheck(stat) \
    { cudnnErrCheck_((stat), __FILE__, __LINE__); }
inline void cudnnErrCheck_(cudnnStatus_t stat, const char *file, int line) {
    if (stat != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "cuDNN Error: %s %s %d\n", cudnnGetErrorString(stat), file, line);
    }
}