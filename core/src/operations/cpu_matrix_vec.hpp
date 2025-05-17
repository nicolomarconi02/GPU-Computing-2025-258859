#pragma once

#include "structures/matrix.hpp"
#include "expected.hpp"

namespace Operations {
enum MultiplicationTypes : uint8_t { Sequential = 0, SIZE };
template <typename indexType, typename dataType>
tl::expected<Matrix<indexType, dataType>, std::string> sequentialMultiplication(
    const Matrix<indexType, dataType>& mat,
    const Matrix<indexType, dataType>& vec) {
  if (!(vec.type & MatrixType_::array)) {
    return tl::make_unexpected("Cannot multiply. Vec must be an array!");
  }
  Matrix<indexType, dataType> res(MatrixType_::array, mat.N_ROWS);
  for (indexType i = 1; i <= mat.N_ROWS; i++) {
    indexType rowStart = mat.csr[i - 1];
    indexType rowEnd = mat.csr[i];

    dataType sum = 0;
    for (indexType j = rowStart; j < rowEnd; j++) {
      sum += mat.values[j] * vec.values[mat.columns[j]];
    }
    res.values[i - 1] = sum;
  }
  return res;
}
}  // namespace Operations
