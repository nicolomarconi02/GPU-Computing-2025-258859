#pragma once

#include <cstdint>
#include <ostream>
typedef int MatrixType;

enum class MatrixType_ {
  matrix = 1 << 0,

  coordinate = 1 << 1,
  array = 1 << 2,
  dense = 1 << 3,
  sparse = 1 << 4,

  complex = 1 << 5,
  real = 1 << 6,
  pattern = 1 << 7,
  integer = 1 << 8,

  symmetric = 1 << 9,
  general = 1 << 10,
  skew = 1 << 11,
  hermitian = 1 << 12
};

template <typename T>
class Matrix {
 public:
  MatrixType type;
  uint32_t* rows;
  uint32_t* columns;
  T* values;

  int N_ROWS;
  int N_COLS;
  int N_ELEM;

  friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
    for (int i = 0; i < mat.N_ELEM; i++) {
      os << mat.rows[i] << " " << mat.columns[i] << " " << mat.values[i]
         << std::endl;
    }
    return os;
  }
};
