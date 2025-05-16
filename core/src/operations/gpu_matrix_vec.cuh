#pragma once
#include <cstdint>
#include <cstdio>

namespace Operations {
template <typename T>
__global__ void parallelMultiplication(int N, uint32_t* csr, uint32_t* columns,
                                       T* values, T* vec, T* res) {
  uint32_t count = 0;
  int beginRow = 0;
  int startIndex = threadIdx.x + (blockDim.x * blockIdx.x) + 1;
  int stride = blockDim.x;
  for (int i = startIndex; i <= N; i += stride) {
    count = csr[i - 1];
    beginRow = count;
    printf("thread: %d, startIndex: %d, i: %d, stride: %d, csr[i-1]: %d\n", threadIdx.x, startIndex, i, stride, csr[i-1]);
    for (; count < beginRow + csr[i] - csr[i - 1]; count++) {
      res[i - 1] += values[count] * vec[columns[count]];
      printf("thread: %d, count: %d, beginRow: %d, i: %d, res[i-1]: %d\n", threadIdx.x, count, beginRow,
             i, res[i - 1]);
    }
  }
}
}  // namespace Operations
