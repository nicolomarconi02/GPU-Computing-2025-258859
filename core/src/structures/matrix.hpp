#pragma once

#include <sys/types.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <cstring>
#include "expected.hpp"

typedef uint16_t MatrixType;

enum MatrixType_ : uint16_t {
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
  hermitian = 1 << 12,
  supported_type = matrix | coordinate | array | dense | sparse | real | integer | general | symmetric | pattern
};

template <typename indexType, typename dataType>
class Matrix {
 public:
  Matrix(int _N_ROWS, int _N_COLS, int _N_ELEM)
      : N_ROWS(_N_ROWS), N_COLS(_N_COLS), N_ELEM(_N_ELEM) {
    rows = (indexType*)malloc(N_ELEM * sizeof(indexType));
    columns = (indexType*)malloc(N_ELEM * sizeof(indexType));
    values = (dataType*)malloc(N_ELEM * sizeof(dataType));
  }

  Matrix(MatrixType _type, int _N_ELEM) : N_ELEM(_N_ELEM) {
    if (_type & MatrixType_::array) {
      type = _type;
      values = (dataType*)calloc(N_ELEM, sizeof(dataType));
    }
  }

  ~Matrix() { freeMatrix(); }
  Matrix(const Matrix<indexType, dataType>& other)
      : N_ROWS(other.N_ROWS),
        N_COLS(other.N_COLS),
        N_ELEM(other.N_ELEM),
        type(other.type) {
    rows = (indexType*)malloc(N_ELEM * sizeof(indexType));
    columns = (indexType*)malloc(N_ELEM * sizeof(indexType));
    values = (dataType*)malloc(N_ELEM * sizeof(dataType));
    std::copy(other.rows, other.rows + N_ELEM, rows);
    std::copy(other.columns, other.columns + N_ELEM, columns);
    std::copy(other.values, other.values + N_ELEM, values);

    if (other.csr) {
      csr = (indexType*)malloc((N_ROWS + 1) * sizeof(indexType));
      std::copy(other.csr, other.csr + N_ROWS + 1, csr);
    } else {
      csr = nullptr;
    }
  }
  Matrix(Matrix<indexType, dataType>&& other) noexcept
      : N_ROWS(other.N_ROWS),
        N_COLS(other.N_COLS),
        N_ELEM(other.N_ELEM),
        type(other.type),
        rows(other.rows),
        columns(other.columns),
        values(other.values),
        csr(other.csr) {
    other.rows = nullptr;
    other.columns = nullptr;
    other.values = nullptr;
    other.csr = nullptr;
  }
  Matrix& operator=(const Matrix<indexType, dataType>& other) {
    if (this != &other) {
      freeMatrix();

      N_ROWS = other.N_ROWS;
      N_COLS = other.N_COLS;
      N_ELEM = other.N_ELEM;
      type = other.type;

      rows = (indexType*)malloc(N_ELEM * sizeof(indexType));
      columns = (indexType*)malloc(N_ELEM * sizeof(indexType));
      values = (dataType*)malloc(N_ELEM * sizeof(dataType));
      std::copy(other.rows, other.rows + N_ELEM, rows);
      std::copy(other.columns, other.columns + N_ELEM, columns);
      std::copy(other.values, other.values + N_ELEM, values);

      if (other.csr) {
        csr = (indexType*)malloc((N_ROWS + 1) * sizeof(indexType));
        std::copy(other.csr, other.csr + N_ROWS + 1, csr);
      } else {
        csr = nullptr;
      }
    }
    return *this;
  }
  Matrix& operator=(Matrix<indexType, dataType>&& other) noexcept {
    if (this != &other) {
      freeMatrix();

      N_ROWS = other.N_ROWS;
      N_COLS = other.N_COLS;
      N_ELEM = other.N_ELEM;
      type = other.type;

      rows = other.rows;
      columns = other.columns;
      values = other.values;
      csr = other.csr;

      other.rows = nullptr;
      other.columns = nullptr;
      other.values = nullptr;
      other.csr = nullptr;
    }
    return *this;
  }
  void freeMatrix() {
    if (columns != nullptr) {
      free(columns);
    }
    if (values != nullptr) {
      free(values);
    }
    if (rows != nullptr) {
      free(rows);
    }
    if (csr != nullptr) {
      free(csr);
    }
  }

  tl::expected<bool, std::string> computeCSR() {
    if (type & MatrixType_::array) {
      return tl::make_unexpected("Cannot compute CSR for an array");
    }
    csr = (indexType*)calloc(N_ROWS + 1, sizeof(indexType));
    if (!csr) {
      return tl::make_unexpected("Memory allocation for CSR failed");
    }

    for (int i = 0; i < N_ELEM; i++) {
      if (rows[i] >= static_cast<indexType>(N_ROWS)) {
        return tl::make_unexpected("Rows index exceed limits");
      }
      csr[rows[i] + 1]++;
    }

    for (int i = 1; i <= N_ROWS; i++) {
      csr[i] += csr[i - 1];
    }

    return true;
  }

  MatrixType type = 0;
  indexType* csr = nullptr;
  indexType* rows = nullptr;
  indexType* columns = nullptr;
  dataType* values = nullptr;

  int N_ROWS = 0;
  int N_COLS = 0;
  int N_ELEM = 0;

  friend std::ostream& operator<<(std::ostream& os, const Matrix<indexType, dataType>& mat) {
    if (mat.type & MatrixType_::array) {
      int total = 0;
      for (int i = 0; i < mat.N_ELEM; i++) {
        os << mat.values[i] << " ";
        total += mat.values[i];
      }
      os << std::endl << "TOTAL " << total << std::endl;
    } else {
      for (int i = 0; i < mat.N_ELEM; i++) {
        os << mat.rows[i] << " " << mat.columns[i] << " " << mat.values[i]
           << std::endl;
      }
      os << "\nCSR\n";
      for (int i = 0; i <= mat.N_ROWS; i++) {
        os << mat.csr[i] << " ";
      }
      os << std::endl;
    }

    return os;
  }
};
