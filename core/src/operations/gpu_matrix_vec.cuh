#pragma once

#include <cstdint>
#include <stdio.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include "utils/cuda_utils.cuh"

namespace Operations {
enum MultiplicationTypes : uint8_t {
  ThreadPerRow = 0,
  ElementWise,
  Warp,
  WarpLoop,
  WarpTiled,
  MergeBased,
  CuSparse,
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
    const dataType* __restrict__ vec, dataType* __restrict__ res,
    indexType* __restrict__ globalRowCounter) {
  const indexType warpSize = 32;
  const indexType lane = threadIdx.x % warpSize;
  const indexType warpIdInBlock = threadIdx.x / warpSize;
  const indexType warpsPerBlock = blockDim.x / warpSize;

  while (true) {
    indexType row;
    if (lane == 0) {
      row = atomicAdd(globalRowCounter, 1);
    }
    row = __shfl_sync(0xFFFFFFFF, row, 0);

    if (row >= N_ROWS) {
      break;  // No more work
    }

    const indexType rowStart = __ldg(&csr[row]);
    const indexType rowEnd = __ldg(&csr[row + 1]);

    dataType sum = 0;
    indexType j = rowStart + lane;

#pragma unroll 4
    for (; j < rowEnd; j += warpSize) {
      const indexType col = __ldg(&columns[j]);
      const dataType val = __ldg(&values[j]);
      const dataType x = __ldg(&vec[col]);
      sum += val * x;
    }

    for (indexType offset = warpSize / 2; offset > 0; offset /= 2) {
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
    const indexType* __restrict__ tileColStart,
    const indexType* __restrict__ tileColEnd,
    const indexType* __restrict__ tileRowBlock) {
  const int warpSize = 32;
  const int tid = threadIdx.x;
  const int warpIdInBlock = tid / warpSize;
  const int lane = tid % warpSize;

  int tileIdx = blockIdx.x;
  int rowBlock = tileRowBlock[tileIdx];
  int blockRowStart = rowBlock * ROWS_PER_BLOCK;
  int row = blockRowStart + warpIdInBlock;
  if (row >= N_ROWS) return;

  extern __shared__ dataType s_vec[];

  indexType colStartBlock = tileColStart[blockIdx.x];
  indexType colEndBlock = tileColEnd[blockIdx.x];
  indexType span = colEndBlock - colStartBlock + 1;

  for (indexType tileOffset = 0; tileOffset < span;
       tileOffset += BLOCK_COL_CHUNK) {
    indexType tileStartCol = colStartBlock + tileOffset;
    indexType tileSize = min((indexType)BLOCK_COL_CHUNK, span - tileOffset);

    for (int c = tid; c < tileSize; c += blockDim.x) {
      s_vec[c] = vec[tileStartCol + c];
    }
    __syncthreads();

    indexType rowStart = csr[row];
    indexType rowEnd = csr[row + 1];

    dataType partialSum = 0;

    for (indexType j = rowStart + lane; j < rowEnd; j += warpSize) {
      indexType col = columns[j];
      if (col >= tileStartCol && col < tileStartCol + tileSize) {
        partialSum += values[j] * s_vec[col - tileStartCol];
      }
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      partialSum += __shfl_down_sync(0xFFFFFFFF, partialSum, offset);
    }

    if (lane == 0) {
      atomicAdd(&res[row], partialSum);
    }
  }
}

template <typename indexType, typename dataType>
void SpMVcuSparse(indexType N_ROWS, indexType N_COLS, indexType N_ELEM,
                  const indexType* csr, const indexType* columns,
                  const dataType* values, const dataType* vec, dataType* res) {
  cusparseHandle_t handle;
  cusparseCreate(&handle);

  cusparseSpMatDescr_t matA;
  cusparseCreateCsr(&matA, N_ROWS, N_COLS, N_ELEM, (void*)csr, (void*)columns,
                    (void*)values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

  cusparseDnVecDescr_t vecX;
  cusparseCreateDnVec(&vecX, N_COLS, (void*)vec, CUDA_R_64F);

  cusparseDnVecDescr_t vecY;
  cusparseCreateDnVec(&vecY, N_ROWS, (void*)res, CUDA_R_64F);

  double alpha = 1.0;
  double beta = 0.0;

  size_t bufferSize = 0;
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          matA, vecX, &beta, vecY, CUDA_R_64F,
                          CUSPARSE_SPMV_CSR_ALG1, &bufferSize);

  void* dBuffer = nullptr;
  cudaMalloc(&dBuffer, bufferSize);

  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
               &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, dBuffer);

  cudaFree(dBuffer);
  cusparseDestroySpMat(matA);
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cusparseDestroy(handle);
}

template <typename indexType, typename dataType>
__global__ void parallelMultiplicationMergeBased(
    indexType N_ROWS, indexType N_ELEM, const indexType* __restrict__ csr,
    const indexType* __restrict__ columns, const dataType* __restrict__ values,
    const dataType* __restrict__ vec, dataType* __restrict__ res) {
  indexType tid = blockIdx.x * blockDim.x + threadIdx.x;
  indexType totalThreads = gridDim.x * blockDim.x;

  indexType nnzPerThread = (N_ELEM + totalThreads - 1) / totalThreads;
  indexType start_nnz = tid * nnzPerThread;
  indexType end_nnz = min(start_nnz + nnzPerThread, N_ELEM);

  if (start_nnz >= N_ELEM) return;

  indexType row = findRowFromCSR(csr, N_ROWS, start_nnz);

  indexType row_start = __ldg(&csr[row]);
  indexType row_end = __ldg(&csr[row + 1]);

  for (indexType k = start_nnz; k < end_nnz; k++) {
    while (k >= row_end) {
      row++;
      row_start = __ldg(&csr[row]);
      row_end = __ldg(&csr[row + 1]);
    }

    indexType col = __ldg(&columns[k]);
    dataType val = __ldg(&values[k]);
    dataType vecVal = __ldg(&vec[col]);
    atomicAdd(&res[row], val * vecVal);
  }
}
}  // namespace Operations
