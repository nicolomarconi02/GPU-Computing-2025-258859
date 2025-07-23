#pragma once

#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << #call << ": "                           \
                << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define COMPUTE_N_BLOCKS(type, size) \
  size < 512 ? 1 : std::min((type)GPU_MAX_N_BLOCKS, (type)size / 512);
#define COMPUTE_N_THREAD(type, size) \
  std::min((type)GPU_MAX_N_THREAD, (type)size / N_BLOCKS);

template <typename indexType>
__device__ __forceinline__ indexType findRowFromCSR(
    const indexType* __restrict__ row_ptr, indexType nrows, indexType k) {
  indexType low = 0;
  indexType high = nrows - 1;
  while (low <= high) {
    indexType mid = (low + high) >> 1;
    indexType start = row_ptr[mid];
    indexType end = row_ptr[mid + 1];
    if (k >= start && k < end) {
      return mid;
    } else if (k < start) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return nrows - 1;
}
