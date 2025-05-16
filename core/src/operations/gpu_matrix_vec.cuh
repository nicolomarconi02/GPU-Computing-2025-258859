#pragma once
#include <cstdint>
#include <cstdio>

namespace Operations {
template <typename indexType, typename dataType>
__global__ void parallelMultiplication(indexType N, indexType* csr,
                                       indexType* columns, dataType* values,
                                       dataType* vec, dataType* res) {
  indexType startIndex = threadIdx.x + (blockDim.x * blockIdx.x) + 1;
  if (startIndex > N) {
    return;
  }
  indexType count = 0;
  indexType beginRow = 0;
  indexType stride = blockDim.x;
  for (indexType i = startIndex; i <= N; i += stride) {
    count = csr[i - 1];
    beginRow = count;
    // printf(
    //     "thread: %d, blockDim: %d, blockIdx: %d, startIndex: %d, i: %d, "
    //     "stride: %d, csr[i-1]: %d\n",
    //     threadIdx.x, blockDim.x, blockIdx.x, startIndex, i, stride, csr[i - 1]);
    for (; count < beginRow + csr[i] - csr[i - 1]; count++) {
      res[i - 1] += values[count] * vec[columns[count]];
      // printf("thread: %d, count: %d, beginRow: %d, i: %d, res[i-1]: %d\n",
      // threadIdx.x, count, beginRow,
      //        i, res[i - 1]);
      printf(
          "thread: %d, blockDim: %d, blockIdx: %d, startIndex: %d, i: %d, "
          "stride: %d, count: %d, csr[i-1]: %d\n",
          threadIdx.x, blockDim.x, blockIdx.x, startIndex, i, stride, count,
          csr[i - 1]);
    }
  }
}
}  // namespace Operations
