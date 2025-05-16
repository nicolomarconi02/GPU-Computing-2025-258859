#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include "profiler/profiler.hpp"
#include "structures/matrix.hpp"
#include "utils/cuda_utils.cuh"
#include "defines.hpp"

namespace Utils {
template <typename indexType, typename dataType>
__global__ void bitonicSortStep(indexType* rows, indexType* cols,
                                dataType* values, indexType distance,
                                indexType sequenceSize) {
  indexType currIndex = threadIdx.x + blockDim.x * blockIdx.x;
  indexType compIndex = currIndex ^ distance;

  if (compIndex > currIndex) {
    // ascending sort
    if ((currIndex & sequenceSize) == 0 &&
        (rows[currIndex] > rows[compIndex])) {
      indexType tmpRow = rows[currIndex];
      indexType tmpCol = cols[currIndex];
      dataType tmpVal = values[currIndex];
      rows[currIndex] = rows[compIndex];
      cols[currIndex] = cols[compIndex];
      values[currIndex] = values[compIndex];
      rows[compIndex] = tmpRow;
      cols[compIndex] = tmpCol;
      values[compIndex] = tmpVal;
    }
    // descending sort
    if ((currIndex & sequenceSize) != 0 &&
        (rows[currIndex] < rows[compIndex])) {
      indexType tmpRow = rows[currIndex];
      indexType tmpCol = cols[currIndex];
      dataType tmpVal = values[currIndex];
      rows[currIndex] = rows[compIndex];
      cols[currIndex] = cols[compIndex];
      values[currIndex] = values[compIndex];
      rows[compIndex] = tmpRow;
      cols[compIndex] = tmpCol;
      values[compIndex] = tmpVal;
    }
  }
}

template <typename indexType, typename dataType>
void parallelSort(Matrix<indexType, dataType>& matrix) {
  ScopeProfiler prof("Parallel Sort");

  // compute nearest larger power of 2
  const indexType size =
      std::pow(2, (indexType)std::ceil(std::log2(matrix.N_ELEM)));
  const indexType paddingSize = size - matrix.N_ELEM;

  indexType *rowsDevice, *colsDevice, *indexPadding;
  dataType *valuesDevice, *valuesPadding;

  CUDA_CHECK(cudaMalloc(&rowsDevice, size * sizeof(indexType)));
  CUDA_CHECK(cudaMalloc(&colsDevice, size * sizeof(indexType)));
  CUDA_CHECK(cudaMalloc(&valuesDevice, size * sizeof(dataType)));

  CUDA_CHECK(cudaMemcpy(rowsDevice, matrix.rows,
                        (matrix.N_ELEM) * sizeof(indexType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(colsDevice, matrix.columns,
                        (matrix.N_ELEM) * sizeof(indexType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(valuesDevice, matrix.values,
                        (matrix.N_ELEM) * sizeof(dataType),
                        cudaMemcpyHostToDevice));

  indexPadding = (indexType*)malloc(paddingSize * sizeof(indexType));
  valuesPadding = (dataType*)malloc(paddingSize * sizeof(dataType));

  for (indexType i = 0; i < paddingSize; i++) {
    indexPadding[i] = std::numeric_limits<indexType>::max();
    valuesPadding[i] = std::numeric_limits<dataType>::max();
  }

  CUDA_CHECK(cudaMemcpy(rowsDevice + matrix.N_ELEM, indexPadding,
                        (paddingSize) * sizeof(indexType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(colsDevice + matrix.N_ELEM, indexPadding,
                        (paddingSize) * sizeof(indexType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(valuesDevice + matrix.N_ELEM, valuesPadding,
                        (paddingSize) * sizeof(indexType),
                        cudaMemcpyHostToDevice));

  free(indexPadding);
  free(valuesPadding);

  const indexType N_BLOCKS = COMPUTE_N_BLOCKS(indexType, size);
  const indexType N_THREAD = COMPUTE_N_THREAD(indexType, size);

  for (indexType sequenceSize = 2; sequenceSize <= size; sequenceSize <<= 1) {
    for (indexType distance = sequenceSize >> 1; distance > 0; distance >>= 1) {
      bitonicSortStep<<<N_BLOCKS, N_THREAD>>>(
          rowsDevice, colsDevice, valuesDevice, distance, sequenceSize);
      cudaDeviceSynchronize();
    }
  }

  CUDA_CHECK(cudaMemcpy(matrix.rows, rowsDevice,
                        (matrix.N_ELEM) * sizeof(indexType),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(matrix.columns, colsDevice,
                        (matrix.N_ELEM) * sizeof(indexType),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(matrix.values, valuesDevice,
                        (matrix.N_ELEM) * sizeof(dataType),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(rowsDevice));
  CUDA_CHECK(cudaFree(colsDevice));
  CUDA_CHECK(cudaFree(valuesDevice));
}

}  // namespace Utils
