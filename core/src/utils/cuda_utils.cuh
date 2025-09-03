#pragma once
#include "structures/matrix.hpp"
#include "structures/block_partition.hpp"

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
  size < 512 ? 1 : std::min((type)GPU_MAX_N_BLOCKS, (type)size / 512);
#define COMPUTE_N_THREAD(type, size) \
  std::min((type)GPU_MAX_N_THREAD, (type)size / N_BLOCKS);

template <typename indexType>
__device__ __forceinline__ indexType findRowFromCSR(
    const indexType* __restrict__ row_ptr, indexType nrows, indexType k) {
  indexType low = 0;
  indexType high = nrows - 1;
  while (low <= high) {
    indexType mid = (low + high) >> 1;
    indexType start = row_ptr[mid];
    indexType end = row_ptr[mid + 1];
    if (k >= start && k < end) {
      return mid;
    } else if (k < start) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return nrows - 1;
}

template <typename indexType, typename dataType>
void precomputePartitions(const Matrix<indexType, dataType>& matrix,
                          indexType numBlocks, indexType blockSize,
                          std::vector<BlockPartition<indexType>>& partitions) {
  partitions.resize(numBlocks);
  indexType totalThreads = numBlocks * blockSize;
  indexType totalPathLen = matrix.N_ROWS + matrix.N_ELEM;

  for (indexType i = 0; i < numBlocks; ++i) {
    indexType blockPathStartDiag =
        (i * blockSize * totalPathLen) / totalThreads;
    indexType blockPathEndDiag =
        ((i + 1) * blockSize * totalPathLen) / totalThreads;

    auto findStart = [&](indexType diagonal, indexType& row, indexType& nnz) {
      indexType low = 0, high = matrix.N_ROWS;
      while (low < high) {
        indexType mid = low + (high - low) / 2;
        if (matrix.csr[mid] > diagonal - mid)
          high = mid;
        else
          low = mid + 1;
      }
      row = low - 1;
      nnz = diagonal - row;
    };

    findStart(blockPathStartDiag, partitions[i].startRow,
              partitions[i].startNnz);
    findStart(blockPathEndDiag, partitions[i].endRow, partitions[i].endNnz);
  }
}
