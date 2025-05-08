#pragma once

#include "structures/matrix.hpp"
#include "expected.hpp"
#include "profiler/profiler.hpp"

namespace Operations {
template <typename T>
tl::expected<T*, std::string> multiplication(const Matrix<T>& mat,
                                             const T* vec) {
  ScopeProfiler prof("multiplication");
  T* res = (T*) calloc(mat.N_ROWS, sizeof(T));
  int count = 0;
  int beginRow = 0;
  for (int i = 1; i <= mat.N_ROWS; i++) {
    beginRow = count;
    for (; count < beginRow + mat.csr[i] - mat.csr[i - 1]; count++) {
      res[i - 1] += mat.values[count] * vec[mat.columns[count]];
    }
  }
  return res;
}
}  // namespace operations
