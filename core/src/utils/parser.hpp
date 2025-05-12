#pragma once

#include <algorithm>
#include <cstdio>
#include "structures/matrix.hpp"
#include "expected.hpp"
#include "utils/sort_matrix.hpp"
#include "profiler/profiler.hpp"

extern "C" {
#include "mmio.h"
}

namespace Utils {

tl::expected<MatrixType, std::string> parseMatrixType(
    const MM_typecode& matcode);

bool checkSupportedMatrixType(const MatrixType type);

template <typename T>
void storeMatrix(FILE* file, Matrix<T>& matrix) {
  int counterElem = 0;
  if (matrix.type & MatrixType_::pattern) {
    for (int i = 0; i < matrix.N_ELEM; i++) {
      int ret = fscanf(file, "%d %d\n", &matrix.rows[i], &matrix.columns[i]);
      if(ret == EOF){
        break;
      }
      matrix.values[i] = 1;
      matrix.rows[i]--;
      matrix.columns[i]--;
      int newIndex = Utils::sortMatrixUntil(matrix, i);
      if (matrix.type & MatrixType_::symmetric &&
          matrix.rows[newIndex] != matrix.columns[newIndex]) {
        matrix.values[i + 1] = matrix.values[newIndex];
        matrix.rows[i + 1] = matrix.columns[newIndex];
        matrix.columns[i + 1] = matrix.rows[newIndex];
        counterElem++;
        i++;
        Utils::sortMatrixUntil(matrix, i);
      }
    }
  }
  else if (matrix.type & MatrixType_::array) {
    for (int i = 0; i < matrix.N_ELEM; i++) {
      int ret = fscanf(file, "%d %lg\n", &matrix.rows[i], &matrix.values[i]);
      if(ret == EOF){
        break;
      }
      matrix.rows[i]--;
      matrix.columns[i] = 0;
      Utils::sortMatrixUntil(matrix, i);
    }
  }
  else if (matrix.type & MatrixType_::real || matrix.type & MatrixType_::integer) {
    for (int i = 0; i < matrix.N_ELEM; i++) {
      int ret = fscanf(file, "%d %d %lg\n", &matrix.rows[i], &matrix.columns[i],
             &matrix.values[i]);
      if(ret == EOF){
        break;
      }
      matrix.rows[i]--;
      matrix.columns[i]--;
      int newIndex = Utils::sortMatrixUntil(matrix, i);
      if (matrix.type & MatrixType_::symmetric &&
          matrix.rows[newIndex] != matrix.columns[newIndex]) {
        matrix.values[i + 1] = matrix.values[newIndex];
        matrix.rows[i + 1] = matrix.columns[newIndex];
        matrix.columns[i + 1] = matrix.rows[newIndex];
        counterElem++;
        i++;
        Utils::sortMatrixUntil(matrix, i);
      }
    }
  }

  if (matrix.type & MatrixType_::symmetric) {
    matrix.N_ELEM = (matrix.N_ELEM / 2) + counterElem;
  }
}

template <typename T>
tl::expected<Matrix<T>, std::string> parseMatrixMarketFile(
    const std::string& path) {
  ScopeProfiler prof("parseMatrixMarketFile");
  FILE* inputFile = fopen(path.c_str(), "r");
  MM_typecode matcode;
  int N_ROWS, N_COLS, N_ELEM = 0;
  if (inputFile == NULL) {
    return tl::make_unexpected("Error opening the file " + path);
  }

  if (mm_read_banner(inputFile, &matcode) != 0) {
    fclose(inputFile);
    return tl::make_unexpected("Error parsing the banner");
  }

  if (mm_read_mtx_crd_size(inputFile, &N_ROWS, &N_COLS, &N_ELEM) != 0) {
    fclose(inputFile);
    return tl::make_unexpected("Error reading the matrix size");
  }

  auto retType = parseMatrixType(matcode);
  if (!retType.has_value()) {
    return tl::make_unexpected(retType.error());
  } else {
    if (retType.value() & MatrixType_::symmetric) {
      N_ELEM *= 2;
    }
  }
  Matrix<T> matrix(N_ROWS, N_COLS, N_ELEM);
  matrix.type = retType.value();

  storeMatrix(inputFile, matrix);

  auto retCSR = matrix.computeCSR();

  if (!retCSR.has_value()) {
    return tl::make_unexpected(retCSR.error());
  }

  fclose(inputFile);
  return matrix;
}

}  // namespace Utils
