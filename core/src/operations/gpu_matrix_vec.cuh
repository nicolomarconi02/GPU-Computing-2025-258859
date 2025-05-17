#pragma once
#include <cstdint>
#include <cstdio>

namespace Operations {
template <typename indexType, typename dataType>
__global__ void parallelMultiplicationThreadPerRow(indexType N, indexType* csr,
                                                   indexType* columns,
                                                   dataType* values,
                                                   dataType* vec,
                                                   dataType* res) {
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

template <typename indexType, typename dataType>
__global__ void parallelMultiplicationElementWise(
    indexType N_ROWS, indexType* csr, indexType* columns, dataType* values,
    dataType* vec, dataType* res) {
  indexType index = threadIdx.x + (blockDim.x * blockIdx.x);
  indexType stride = blockDim.x * gridDim.x;

  const indexType NNZ = csr[N_ROWS];

  for (; index < NNZ; index += stride) {
    indexType lhs = 0;
    indexType rhs = N_ROWS;
    while (lhs < rhs) {
      indexType mid = (lhs + rhs) / 2;
      if (csr[mid + 1] <= index) {
        lhs = mid + 1;
      } else {
        rhs = mid;
      }
    }

    indexType rowIndex = rhs;
    dataType product = values[index] * vec[columns[index]];
    atomicAdd(&res[rowIndex], product);
  }
}
}  // namespace Operations
