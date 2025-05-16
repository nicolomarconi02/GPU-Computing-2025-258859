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
  size < 512 ? 1 : std::min((type) GPU_MAX_N_BLOCKS, (type) size / 512);
#define COMPUTE_N_THREAD(type, size) std::min((type) GPU_MAX_N_THREAD, (type) size / N_BLOCKS);
