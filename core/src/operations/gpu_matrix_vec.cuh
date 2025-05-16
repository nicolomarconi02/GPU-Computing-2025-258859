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
  indexType stride = blockDim.x * gridDim.x;
  for (indexType i = startIndex; i <= N; i += stride) {
    indexType rowStart = csr[i - 1];
    indexType rowEnd = csr[i];

    dataType sum = 0;
    for (indexType j = rowStart; j < rowEnd; j++) {
      sum += values[j] * vec[columns[j]];
    }
    res[i - 1] = sum;
  }
}
}  // namespace Operations
