#pragma once

#include <cstdint>
namespace Operations {
enum MultiplicationTypes : uint8_t {
  ThreadPerRow = 0,
  ElementWise = 1,
  Warp = 2,
  SIZE
};
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

template <typename indexType, typename dataType>
__global__ void parallelMultiplicationWarp(const indexType N_ROWS, const indexType* __restrict__ csr,
                              const indexType* __restrict__ columns, const dataType* __restrict__ values,
                              const dataType* __restrict__ vec, dataType* __restrict__ res) {
  const int warpSize = 32;
  const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int warp_id = thread_id / warpSize;
  const int lane = threadIdx.x % warpSize; 

  if (warp_id >= N_ROWS) return;

  indexType row = warp_id;
  indexType row_start = csr[row];
  indexType row_end = csr[row + 1];

  dataType sum = 0;

  for (indexType j = row_start + lane; j < row_end; j += warpSize) {
    sum += values[j] * vec[columns[j]];
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }

  if (lane == 0) {
    res[row] = sum;
  }
}

}  // namespace Operations
