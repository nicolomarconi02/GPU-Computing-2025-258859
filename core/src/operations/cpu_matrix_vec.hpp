#pragma once

#include "structures/matrix.hpp"
#include "expected.hpp"
#include "profiler/profiler.hpp"

namespace Operations {
enum MultiplicationTypes : uint8_t {
  Sequential = 0,
  SIZE
};
template <typename indexType, typename dataType>
tl::expected<Matrix<indexType, dataType>, std::string> sequentialMultiplication(const Matrix<indexType, dataType>& mat,
                                             const Matrix<indexType, dataType>& vec) {
  if(!(vec.type & MatrixType_::array)){
    return tl::make_unexpected("Cannot multiply. Vec must be an array!");
  }
  Matrix<indexType, dataType> res(MatrixType_::array, mat.N_ROWS);
  int count = 0;
  int beginRow = 0;
  for (int i = 1; i <= mat.N_ROWS; i++) {
    beginRow = count;
    for (; count < beginRow + mat.csr[i] - mat.csr[i - 1]; count++) {
      res.values[i - 1] += mat.values[count] * vec.values[mat.columns[count]];
    }
  }
  return res;
}
}  // namespace operations
