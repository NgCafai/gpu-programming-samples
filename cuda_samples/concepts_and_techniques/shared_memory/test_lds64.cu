#include <cstdint>

__global__ void smem_1(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[tid];
}


__global__ void smem_2(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[tid / 2];
}



int main() {
  uint32_t *d_a;
  cudaMalloc(&d_a, sizeof(uint32_t) * 128);
//   smem_1<<<1, 32>>>(d_a);
  smem_2<<<1, 32>>>(d_a);
  cudaFree(d_a);
  cudaDeviceSynchronize();
  return 0;
}