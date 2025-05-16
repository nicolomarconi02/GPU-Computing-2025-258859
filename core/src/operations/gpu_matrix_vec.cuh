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
  indexType stride = blockDim.x * gridDim.x;
  for (indexType i = startIndex; i <= N; i += stride) {
    count = csr[i - 1];
    beginRow = count;
    for (; count < beginRow + csr[i] - csr[i - 1]; count++) {
      res[i - 1] += values[count] * vec[columns[count]];
    }
  }
}
}  // namespace Operations
