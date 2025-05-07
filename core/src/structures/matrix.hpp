#pragma once

#include <sys/types.h>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include "expected.hpp"

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
  Matrix(int _N_ROWS, int _N_COLS, int _N_ELEM)
      : N_ROWS(_N_ROWS), N_COLS(_N_COLS), N_ELEM(_N_ELEM) {
    rows = (uint32_t*)malloc(N_ELEM * sizeof(uint32_t));
    columns = (uint32_t*)malloc(N_ELEM * sizeof(uint32_t));
    values = (T*)malloc(N_ELEM * sizeof(T));
  }

  void freeMatrix() {
    free(rows);
    free(columns);
    free(values);
    if (csr != nullptr) {
      free(csr);
    }
  }

  tl::expected<bool, std::string> computeCSR() {
    csr = (uint32_t*)calloc(N_ROWS + 1, sizeof(uint32_t));
    if (!csr) {
      return tl::make_unexpected("Memory allocation for CSR failed");
    }

    for (int i = 0; i < N_ELEM; i++) {
      if (rows[i] >= static_cast<uint32_t>(N_ROWS)) {
        return tl::make_unexpected("Rows index exceed limits");
      }
      csr[rows[i] + 1]++;
    }

    for (int i = 1; i <= N_ROWS; i++) {
      csr[i] += csr[i - 1];
    }

    return true;
  }

  MatrixType type;
  uint32_t* csr;
  uint32_t* rows;
  uint32_t* columns;
  T* values;

  int N_ROWS;
  int N_COLS;
  int N_ELEM;

  Matrix<T>& operator=(Matrix<T> other) {
    this->N_ELEM = other.N_ELEM;
    this->N_COLS = other.N_COLS;
    this->N_ROWS = other.N_ROWS;
    this->type = other.type;
    this->values = std::move(other.values);
    this->columns = std::move(other.columns);
    this->rows = std::move(other.rows);
    this->csr = std::move(other.csr);
  }

  friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
    for (int i = 0; i < mat.N_ELEM; i++) {
      os << mat.rows[i] << " " << mat.columns[i] << " " << mat.values[i]
         << std::endl;
    }
    os << "\nCSR\n";
    for (int i = 0; i <= mat.N_ROWS; i++) {
      os << mat.csr[i] << " ";
    }
    os << std::endl;
    return os;
  }
};
