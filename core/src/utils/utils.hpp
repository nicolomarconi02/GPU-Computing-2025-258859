#pragma once

#include <defines.hpp>
#include <expected.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "structures/matrix.hpp"
#include "profiler/profiler.hpp"

namespace Utils {

template <typename dataType>
void printVector(const dataType* vec, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << vec[i] << " ";
  }
  std::cout << std::endl;
}

template <typename indexType, typename dataType>
tl::expected<bool, std::string> saveResultsToFile(
    Matrix<indexType, dataType>& matrix, Matrix<indexType, dataType>& vector,
    Matrix<indexType, dataType>& result) {
  if (!std::filesystem::exists(MATRICES_OUTPUT_PATH)) {
    std::filesystem::create_directory(MATRICES_OUTPUT_PATH);
  }
  std::string fileName = "output-" + formatDate(getTimestampMicroseconds()) +
                         "-" + formatTime(getTimestampMicroseconds());

  std::ofstream outputFile(MATRICES_OUTPUT_PATH + fileName, std::ofstream::out);
  if (!outputFile.is_open()) {
    return tl::make_unexpected("Error while opening output file");
  }
  outputFile << "INPUT MATRIX" << std::endl << matrix << std::endl;
  outputFile << "_________________________________________________"
             << std::endl;
  outputFile << "INPUT VECTOR" << std::endl << vector << std::endl;
  outputFile << "_________________________________________________"
             << std::endl;
  outputFile << "OUTPUT VECTOR" << std::endl << result << std::endl;
  return true;
}

template <typename indexType>
void computeBlockColRanges(indexType N_ROWS, const indexType* csr,
                           const indexType* columns,
                           const indexType ROWS_PER_BLOCK,
                           const indexType BLOCK_COL_CHUNK,
                           std::vector<indexType>& colStartOut,
                           std::vector<indexType>& colEndOut,
                           std::vector<indexType>& tileCountOut) {
  indexType numBlocks = (N_ROWS + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  colStartOut.resize(numBlocks);
  colEndOut.resize(numBlocks);
  tileCountOut.resize(numBlocks);

  for (indexType blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
    indexType rowStart = blockIndex * ROWS_PER_BLOCK;
    indexType rowEnd = std::min(N_ROWS, rowStart + ROWS_PER_BLOCK);

    indexType minCol = std::numeric_limits<indexType>::max();
    indexType maxCol = 0;

    for (indexType r = rowStart; r < rowEnd; ++r) {
      for (indexType j = csr[r]; j < csr[r + 1]; ++j) {
        indexType c = columns[j];
        if (c < minCol) minCol = c;
        if (c > maxCol) maxCol = c;
      }
    }

    if (minCol == std::numeric_limits<indexType>::max()) {
      minCol = 0;
      maxCol = 0;
    }


    indexType span = maxCol - minCol + 1;
    colStartOut[blockIndex] = minCol;
    colEndOut[blockIndex] = maxCol;

    tileCountOut[blockIndex] =
        (indexType)((span + BLOCK_COL_CHUNK - 1) / BLOCK_COL_CHUNK);
  }
}
}  // namespace Utils
