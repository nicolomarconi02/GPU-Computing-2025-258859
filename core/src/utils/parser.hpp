#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include "structures/matrix.hpp"
#include "expected.hpp"
#include "utils/sort_matrix.hpp"
#include "profiler/profiler.hpp"
#include "defines.hpp"

extern "C" {
#include "mmio.h"
}

namespace Utils {
 
/// Parse the Matrix Market type and map it to an internal MatrixType.
/// @param matcode The Matrix Market typecode.
/// @return An expected containing the internal MatrixType or an error string.
tl::expected<MatrixType, std::string> parseMatrixType(
    const MM_typecode& matcode);

/// Check whether a given MatrixType is supported.
/// @param type MatrixType enum to validate.
/// @return true if supported, false otherwise.
bool checkSupportedMatrixType(const MatrixType type);

/// Read matrix elements from a Matrix Market file and store them in a Matrix structure.
/// This function supports pattern, array, real, and integer matrices, and handles
/// symmetric matrices by duplicating off-diagonal entries.
/// @tparam indexType The type used for row/column indices.
/// @tparam dataType The type used for matrix values.
/// @param file Pointer to the already opened Matrix Market file.
/// @param matrix Matrix structure to be filled.
template <typename indexType, typename dataType>
void storeMatrix(FILE* file, Matrix<indexType, dataType>& matrix) {
  int counterElem = 0;
  if (matrix.type & MatrixType_::pattern) {
    for (int i = 0; i < matrix.N_ELEM; i++) {
      if (fscanf(file, "%d %d\n", &matrix.rows[i], &matrix.columns[i]) == EOF) {
        break;
      }

      int newIndex = i;
      matrix.values[i] = 1;
      matrix.rows[i]--;
      matrix.columns[i]--;

      if (matrix.type & MatrixType_::symmetric &&
          matrix.rows[newIndex] != matrix.columns[newIndex]) {
        matrix.values[i + 1] = matrix.values[newIndex];
        matrix.rows[i + 1] = matrix.columns[newIndex];
        matrix.columns[i + 1] = matrix.rows[newIndex];
        counterElem++;
        i++;
      }
    }
  } else if (matrix.type & MatrixType_::array) {
    for (int i = 0; i < matrix.N_ELEM; i++) {
      if (fscanf(file, "%d %lg\n", &matrix.rows[i], &matrix.values[i]) == EOF) {
        break;
      }
      matrix.rows[i]--;
      matrix.columns[i] = 0;
    }
  } else if (matrix.type & MatrixType_::real ||
             matrix.type & MatrixType_::integer) {
    for (int i = 0; i < matrix.N_ELEM; i++) {
      if (fscanf(file, "%d %d %lg\n", &matrix.rows[i], &matrix.columns[i],
                 &matrix.values[i]) == EOF) {
        break;
      }
      int newIndex = i;
      matrix.rows[i]--;
      matrix.columns[i]--;
      if (matrix.type & MatrixType_::symmetric &&
          matrix.rows[newIndex] != matrix.columns[newIndex]) {
        matrix.values[i + 1] = matrix.values[newIndex];
        matrix.rows[i + 1] = matrix.columns[newIndex];
        matrix.columns[i + 1] = matrix.rows[newIndex];
        counterElem++;
        i++;
      }
    }
  }

  if (matrix.type & MatrixType_::symmetric) {
    matrix.N_ELEM = (matrix.N_ELEM / 2) + counterElem;
  }
}

/// Parse a Matrix Market file and convert it into a Matrix object.
/// The function supports symmetric matrices and optionally sorts and computes CSR if in CPU mode.
/// @tparam indexType The type used for row/column indices.
/// @tparam dataType The type used for matrix values.
/// @param path Path to the Matrix Market (.mtx) file.
/// @return An expected containing the parsed Matrix or an error string.
template <typename indexType, typename dataType>
tl::expected<Matrix<indexType, dataType>, std::string> parseMatrixMarketFile(
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
  Matrix<indexType, dataType> matrix(N_ROWS, N_COLS, N_ELEM);
  matrix.type = retType.value();

  {
    ScopeProfiler store("storeMatrix");
    storeMatrix(inputFile, matrix);
  }

  if (executionMode == Mode_::CPU) {
    {
      ScopeProfiler sort("bitonicSort");
      Utils::cpuBitonicSort(matrix);
    }
    auto retCSR = matrix.computeCSR();

    if (!retCSR.has_value()) {
      return tl::make_unexpected(retCSR.error());
    }
  }
  fclose(inputFile);
  return matrix;
}

}  // namespace Utils
