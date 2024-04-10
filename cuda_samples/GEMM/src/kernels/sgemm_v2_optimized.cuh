#pragma once

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(data) (reinterpret_cast<float4 *>(&(data))[0])

template <const int BLOCK_ITEMS_M,  // Height in rows of a block-wide tile in matrix C
          const int BLOCK_ITEMS_N,  // Width in columns of a block-wide tile in matrix C
          const int BLOCK_ITEMS_K,  // Width in columns of a block-wide tile in matrix A and \
                                  height in rows of a block-wide tile in matrix B
          const int THREAD_ITEMS_M, // Height in rows of a thread tile in C
          const int THREAD_ITEMS_N  // Width in columns of a thread tile in C
          >
__global__ void SgemmV2(int M, int N, int K, float *A, float *B, float *C)
{
    // block index - use 2D gridDim
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // thread index - use 1D blockDim, and map threads to correpsonding data positions
    const int tid = threadIdx.x;

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int THREAD_NUM_PER_BLOCK = (BLOCK_ITEMS_M / THREAD_ITEMS_M) * (BLOCK_ITEMS_N / THREAD_ITEMS_N);

    // map the threads in a warp tile to a 4x8 layout
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    // map the warps in a block tile to a Nx2 layout
    const int A_smem_load_offset = (warp_id / 2) * 32 + mma_tid_y * 4;
    const int B_smem_load_offset = (warp_id % 2) * 64 + mma_tid_x * 4;

    // allocate shared memory for block tile
    __shared__ float A_smem[2][BLOCK_ITEMS_K][BLOCK_ITEMS_M];
    __shared__ float B_smem[2][BLOCK_ITEMS_K][BLOCK_ITEMS_N];

    // allocate register for A and B for thread tile
    float A_frag[2][THREAD_ITEMS_M];
    float B_frag[2][THREAD_ITEMS_N];
    // allocate register for C for thread tile
    float accum[THREAD_ITEMS_M][THREAD_ITEMS_N] = {0.0f};

    // -----------------------------------------------------------------------------
    // calculate the indices that this thread will load from globa memory into shared
    // memory, and each thread will load 4 elements(16 bytes) at each step

    // allocate register used to load global memory
    // Note: each thread will load data in two steps:
    //     1) load from global mem into registers; 2) store into shared memory.
    // This is how the data is actually loaded from gmem to smem in Architecutre before
    // Ampere. We just write the process explicitly to utilize ILP(Instruction Level
    // Parallelism). In Ampere and later architectures, we can use cuda::memcpy_async().
    //
    // The size is the number of elements that each thread will load for one <block tile>.
    float A_ldg_reg[BLOCK_ITEMS_M * BLOCK_ITEMS_K / THREAD_NUM_PER_BLOCK];
    float B_ldg_reg[BLOCK_ITEMS_K * BLOCK_ITEMS_N / THREAD_NUM_PER_BLOCK];

    // thread number required to load data for one row of <block tile> of A and B
    constexpr int A_TILE_THREAD_PER_ROW = BLOCK_ITEMS_K / 4;
    constexpr int B_TILE_THREAD_PER_ROW = BLOCK_ITEMS_N / 4;

    // row and col index in <block tile> of A and B that this thread is responsible for
    const int A_tile_row_start = tid / A_TILE_THREAD_PER_ROW;
    const int A_tile_col_start = (tid % A_TILE_THREAD_PER_ROW) * 4;
    const int B_tile_row_start = tid / B_TILE_THREAD_PER_ROW;
    const int B_tile_col_start = (tid % B_TILE_THREAD_PER_ROW) * 4;

    // if each thread needs to load more than 4 elements, then stride is used
    // to move to the next target row
    constexpr int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    constexpr int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // -----------------------------------------------------------------------------
    // main logic

    // move block tile to beginning of A's row and B's column
    A += by * BLOCK_ITEMS_M * K;
    B += bx * BLOCK_ITEMS_N;

    // --------------------------------------------------------
    // load first block tile from global mem to shared mem
    // load into A_smem
#pragma unroll
    for (int offset = 0; offset + A_TILE_ROW_STRIDE <= BLOCK_ITEMS_M; offset += A_TILE_ROW_STRIDE)
    {
        int ldg_reg_idx = offset / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(A_ldg_reg[ldg_reg_idx]) =
            FETCH_FLOAT4(A[OFFSET(A_tile_row_start + offset, // row
                                  A_tile_col_start,          // col
                                  K)]);
        // transpose the data
        A_smem[0][A_tile_col_start][A_tile_row_start + offset] = A_ldg_reg[ldg_reg_idx];
        A_smem[0][A_tile_col_start + 1][A_tile_row_start + offset] = A_ldg_reg[ldg_reg_idx + 1];
        A_smem[0][A_tile_col_start + 2][A_tile_row_start + offset] = A_ldg_reg[ldg_reg_idx + 2];
        A_smem[0][A_tile_col_start + 3][A_tile_row_start + offset] = A_ldg_reg[ldg_reg_idx + 3];
    }

    // load into B_smem
#pragma unroll
    for (int offset = 0; offset + B_TILE_ROW_STRIDE <= BLOCK_ITEMS_K; offset += B_TILE_ROW_STRIDE)
    {
        FETCH_FLOAT4(B_smem[B_tile_row_start + offset][B_tile_col_start]) =
            FETCH_FLOAT4(B[OFFSET(B_tile_row_start + offset, // row
                                  B_tile_col_start,          // col
                                  N)]);
    }
    __syncthreads();

    // --------------------------------------------------------
    // load first fragment from shared memory to register for computation
    // Note: the thread layout in a warp is 4x8
    FETCH_FLOAT4(A_frag[0][0]) = FETCH_FLOAT4(A_smem[0][0][A_smem_load_offset]);
    FETCH_FLOAT4(A_frag[0][4]) = FETCH_FLOAT4(A_smem[0][0][A_smem_load_offset + 4 * 4]);

    FETCH_FLOAT4(B_frag[0][0]) = FETCH_FLOAT4(B_smem[0][0][B_smem_load_offset]);
    FETCH_FLOAT4(B_frag[0][4]) = FETCH_FLOAT4(B_smem[0][0][B_smem_load_offset + 8 * 4]);

    int write_stage_idx = 1;
    int block_tile_idx = 0;
    do
    {
        // --------------------------------------------------------
        // load next block tile from global mem to register,
        // which will be stored into shared mem later

        // next block tile index
        block_tile_idx += BLOCK_ITEMS_K;

        if (block_tile_idx < K)
        {
            // load into A_ldg_reg
#pragma unroll
            for (int offset = 0; offset + A_TILE_ROW_STRIDE <= BLOCK_ITEMS_M;
                 offset += A_TILE_ROW_STRIDE)
            {
                int ldg_reg_idx = offset / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(A_ldg_reg[ldg_reg_idx]) =
                    FETCH_FLOAT4(A[OFFSET(A_tile_row_start + offset,         // row
                                          A_tile_col_start + block_tile_idx, // col
                                          K)]);
            }

            // load into B_ldg_reg
#pragma unroll
            for (int offset = 0; offset + B_TILE_ROW_STRIDE <= BLOCK_ITEMS_K;
                 offset += B_TILE_ROW_STRIDE)
            {
                int ldg_reg_idx = offset / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(B_ldg_reg[ldg_reg_idx]) =
                    FETCH_FLOAT4(B[OFFSET(block_tile_idx + B_tile_row_start + offset, // row
                                          B_tile_col_start,                           // col
                                          N)]);
            }
        }

        // --------------------------------------------------------
        // read data from current block tile(already on shared memory) to register for computation

        int read_stage_idx = write_stage_idx ^ 1;
#pragma unroll
        for (int frag_k = 0; frag_k < BLOCK_ITEMS_K - 1; ++frag_k)
        {
            // load next fragment from shared memory to register
            // load A fragment
            FETCH_FLOAT4(A_frag[(frag_k + 1) % 2][0]) =
                FETCH_FLOAT4(A_smem[read_stage_idx][frag_k + 1][A_smem_load_offset]);
            FETCH_FLOAT4(A_frag[(frag_k + 1) % 2][4]) =
                FETCH_FLOAT4(A_smem[read_stage_idx][frag_k + 1][A_smem_load_offset + 4 * 4]);
            // load B fragment
            FETCH_FLOAT4(B_frag[(frag_k + 1) % 2][0]) =
                FETCH_FLOAT4(B_smem[read_stage_idx][frag_k + 1][B_smem_load_offset]);
            FETCH_FLOAT4(B_frag[(frag_k + 1) % 2][4]) =
                FETCH_FLOAT4(B_smem[read_stage_idx][frag_k + 1][B_smem_load_offset + 8 * 4]);

            // compute on current fragment in a outter product way
#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_ITEMS_M; ++thread_y)
            {
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_ITEMS_N; ++thread_x)
                {
                    accum[thread_y][thread_x] +=
                        A_frag[frag_k % 2][thread_y] * B_frag[frag_k % 2][thread_x];
                }
            }
        }

        // --------------------------------------------------------
        // load the next block tile from register to shared memory
        if (block_tile_idx < K)
        {
            // load into A_smem[write_stage_idx]
#pragma unroll
            for (int offset = 0; offset + A_TILE_ROW_STRIDE <= BLOCK_ITEMS_M;
                 offset += A_TILE_ROW_STRIDE)
            {
                int ldg_reg_idx = offset / A_TILE_ROW_STRIDE * 4;
                A_smem[write_stage_idx][A_tile_col_start][A_tile_row_start + offset] =
                    A_ldg_reg[ldg_reg_idx];
                A_smem[write_stage_idx][A_tile_col_start + 1][A_tile_row_start + offset] =
                    A_ldg_reg[ldg_reg_idx + 1];
                A_smem[write_stage_idx][A_tile_col_start + 2][A_tile_row_start + offset] =
                    A_ldg_reg[ldg_reg_idx + 2];
                A_smem[write_stage_idx][A_tile_col_start + 3][A_tile_row_start + offset] =
                    A_ldg_reg[ldg_reg_idx + 3];
            }

            // load into B_smem[write_stage_idx]
#pragma unroll
            for (int offset = 0; offset + B_TILE_ROW_STRIDE <= BLOCK_ITEMS_K;
                 offset += B_TILE_ROW_STRIDE)
            {
                int ldg_reg_idx = offset / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(B_smem[write_stage_idx][B_tile_row_start + offset][B_tile_col_start]) =
                    FETCH_FLOAT4(B_ldg_reg[ldg_reg_idx]);
            }

            // use double buffer, only need one sync
            __syncthreads();
            // switch read and write stage
            write_stage_idx ^= 1;
        }

        // --------------------------------------------------------
        // load the first fragment of next block tile from shared memory to register
        FETCH_FLOAT4(A_frag[BLOCK_ITEMS_K % 2][0]) =
            FETCH_FLOAT4(A_smem[read_stage_idx ^ 1][0][A_smem_load_offset]);
        FETCH_FLOAT4(A_frag[BLOCK_ITEMS_K % 2][4]) =
            FETCH_FLOAT4(A_smem[read_stage_idx ^ 1][0][A_smem_load_offset + 4 * 4]);
        FETCH_FLOAT4(B_frag[BLOCK_ITEMS_K % 2][0]) =
            FETCH_FLOAT4(B_smem[read_stage_idx ^ 1][0][B_smem_load_offset]);
        FETCH_FLOAT4(B_frag[BLOCK_ITEMS_K % 2][4]) =
            FETCH_FLOAT4(B_smem[read_stage_idx ^ 1][0][B_smem_load_offset + 8 * 4]);

        // --------------------------------------------------------
        // compute on the last fragment of current block tile
#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_ITEMS_M; ++thread_y)
        {
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_ITEMS_N; ++thread_x)
            {
                accum[thread_y][thread_x] += A_frag[(BLOCK_ITEMS_K - 1) % 2][thread_y] *
                                             B_frag[(BLOCK_ITEMS_K - 1) % 2][thread_x];
            }
        }

    } while (block_tile_idx < K);

    // --------------------------------------------------------
    // store the result to global memory
    // that is, store the 4 4x4 thread sub tiles in accum to C
    int C_row_start = by * BLOCK_ITEMS_M + A_smem_load_offset;
    int C_col_start = bx * BLOCK_ITEMS_N + B_smem_load_offset;
    C += OFFSET(C_row_start, C_col_start, N);

    for (int idx_m = 0; idx_m < THREAD_ITEMS_M; idx_m += 4)
    {
        for (int idx_n = 0; idx_n < THREAD_ITEMS_N; idx_n += 4)
        {
            for (int i = 0; i < 4; i++)
            {
                FETCH_FLOAT4(C[OFFSET(idx_m * 4 + i, // row
                                      idx_n * 8,     // col
                                      N)]) = FETCH_FLOAT4(accum[idx_m + i][idx_n]);
            }
        }
    }

    // Done!!!
}