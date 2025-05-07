#pragma once

#include <cstdio>
#include <iostream>
#include "structures/matrix.hpp"
#include "expected.hpp"
#include "utils/sort_matrix.hpp"

extern "C" {
#include "mmio.h"
}

namespace Utils {

template <typename T>
tl::expected<Matrix<T>, std::string> parseMatrixMarketFile(const std::string& path) {
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

  if(mm_read_mtx_crd_size(inputFile, &N_ROWS, &N_COLS, &N_ELEM) != 0){
    fclose(inputFile);
    return tl::make_unexpected("Error reading the matrix size");
  }

  Matrix<T> matrix(N_ROWS, N_COLS, N_ELEM);
  for(int i = 0; i < matrix.N_ELEM; i++){
    fscanf(inputFile, "%d %d %lg\n", &matrix.rows[i], &matrix.columns[i], &matrix.values[i]);
    matrix.rows[i]--;
    matrix.columns[i]--;
  }
 
  sortMatrix(matrix); 

  fclose(inputFile);
  return matrix;
}

}  // namespace Utils
