#pragma once

#include <cstdint>
#include <stdio.h>
namespace Operations {
enum MultiplicationTypes : uint8_t {
  ThreadPerRow = 0,
  ElementWise,
  Warp,
  WarpLoop,
  WarpTiled,
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
__global__ void parallelMultiplicationWarp(
    const indexType N_ROWS, const indexType* __restrict__ csr,
    const indexType* __restrict__ columns, const dataType* __restrict__ values,
    const dataType* __restrict__ vec, dataType* __restrict__ res) {
  const int warpSize = 32;
  const int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  const int warpId = threadId / warpSize;
  const int lane = threadIdx.x % warpSize;

  if (warpId >= N_ROWS) return;

  indexType row = warpId;
  indexType rowStart = csr[row];
  indexType rowEnd = csr[row + 1];

  dataType sum = 0;

  for (indexType j = rowStart + lane; j < rowEnd; j += warpSize) {
    sum += values[j] * vec[columns[j]];
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }

  if (lane == 0) {
    res[row] = sum;
  }
}

template <typename indexType, typename dataType>
__global__ void parallelMultiplicationWarpLoop(
    const indexType N_ROWS, const indexType* __restrict__ csr,
    const indexType* __restrict__ columns, const dataType* __restrict__ values,
    const dataType* __restrict__ vec, dataType* __restrict__ res) {
  const int warpSize = 32;
  const int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  const int warpId = threadId / warpSize;
  const int lane = threadIdx.x % warpSize;
  const int totalWarps = gridDim.x * blockDim.x / warpSize;

  if (warpId >= N_ROWS) return;

  for (indexType row = warpId; row < N_ROWS; row += totalWarps) {
    indexType rowStart = csr[row];
    indexType rowEnd = csr[row + 1];

    dataType sum = 0;

    for (indexType j = rowStart + lane; j < rowEnd; j += warpSize) {
      sum += values[j] * vec[columns[j]];
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane == 0) {
      res[row] = sum;
    }
  }
}

template <typename indexType, typename dataType, int ROWS_PER_BLOCK,
          int BLOCK_COL_CHUNK>
__global__ void parallelMultiplicationWarpTiled(
    indexType N_ROWS, const indexType* __restrict__ csr,
    const indexType* __restrict__ columns, const dataType* __restrict__ values,
    const dataType* __restrict__ vec, dataType* __restrict__ res,
    const indexType* __restrict__ block_col_start,
    const indexType* __restrict__ block_col_end) {
  const int warpSize = 32;
  const int tid = threadIdx.x;
  const int warp_id_in_block = tid / warpSize;
  const int lane = tid % warpSize;

  int block_row_start = blockIdx.x * ROWS_PER_BLOCK;
  int row = block_row_start + warp_id_in_block;
  if (row >= N_ROWS) return;

  extern __shared__ dataType s_vec[];

  indexType col_start_block = block_col_start[blockIdx.x];
  indexType col_end_block = block_col_end[blockIdx.x];
  indexType span = col_end_block - col_start_block + 1;

  dataType sum = 0;

  for (indexType tile_offset = 0; tile_offset < span;
       tile_offset += BLOCK_COL_CHUNK) {
    indexType tile_start_col = col_start_block + tile_offset;
    indexType tile_size = min((indexType)BLOCK_COL_CHUNK, span - tile_offset);

    for (int c = tid; c < tile_size; c += blockDim.x) {
      s_vec[c] = vec[tile_start_col + c];
    }
    __syncthreads();

    indexType row_start = csr[row];
    indexType row_end = csr[row + 1];

    for (indexType j = row_start + lane; j < row_end; j += warpSize) {
      indexType col = columns[j];
      if (col >= tile_start_col && col < tile_start_col + tile_size) {
        sum += values[j] * s_vec[col - tile_start_col];
      }
    }

    __syncthreads();
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }

  if (lane == 0) {
    res[row] = sum;
  }
}
}  // namespace Operations
