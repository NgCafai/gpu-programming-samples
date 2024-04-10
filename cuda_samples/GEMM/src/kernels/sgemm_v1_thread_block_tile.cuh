#pragma once

/*
 * C = A * B, row-major
 *
 * Template parameters:
 *     BLOCK_TILE_SIZE: the height and width of the block tile
 *
 * Parameters:
 *     A: the input matrix A of shape (M, K), row-major
 *     B: the input matrix B of shape (K, N), row-major
 *     C: the output matrix C of shape (M, N), row-major
 */
template <const int BLOCK_TILE_SIZE>
__global__ void SgemmV1(int M, int N, int K, const float *A, const float *B, float *C)
{
    __shared__ float As[BLOCK_TILE_SIZE][BLOCK_TILE_SIZE];
    __shared__ float Bs[BLOCK_TILE_SIZE][BLOCK_TILE_SIZE];

    // block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_c = by * BLOCK_TILE_SIZE + ty;
    int col_c = bx * BLOCK_TILE_SIZE + tx;

    int row_a = row_c;
    int col_b = col_c;

    int col_a, row_b;
    float dot_sum = 0;
    for (int i = 0; i < K; i += BLOCK_TILE_SIZE)
    {
        // load data to shared memory
        col_a = i + tx;
        row_b = i + ty;

        if (row_a < M && col_a < K)
        {
            As[ty][tx] = A[row_a * K + col_a];
        }
        else
        {
            As[ty][tx] = 0;
        }

        if (row_b < K && col_b < N)
        {
            Bs[ty][tx] = B[row_b * N + col_b];
        }
        else
        {
            Bs[ty][tx] = 0;
        }

        __syncthreads();

        // do dot product
        for (int inner_idx = 0; inner_idx < BLOCK_TILE_SIZE; ++inner_idx)
        {
            dot_sum += As[ty][inner_idx] * Bs[inner_idx][tx];
        }
        __syncthreads();
    }

    if (row_c < M && col_c < N)
    {
        C[row_c * N + col_c] = dot_sum;
    }
}