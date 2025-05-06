#pragma once

#include <cstdint>
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
  if (inputFile == NULL) {
    return tl::make_unexpected("Error opening the file " + path);
  }
  Matrix<T> matrix;

  if (mm_read_banner(inputFile, &matcode) != 0) {
    fclose(inputFile);
    return tl::make_unexpected("Error parsing the banner");
  }

  if(mm_read_mtx_crd_size(inputFile, &matrix.N_ROWS, &matrix.N_COLS, &matrix.N_ELEM) != 0){
    fclose(inputFile);
    return tl::make_unexpected("Error reading the matrix size");
  }

  matrix.rows = (uint32_t *) malloc(matrix.N_ELEM * sizeof(uint32_t));
  matrix.columns = (uint32_t *) malloc(matrix.N_ELEM * sizeof(uint32_t));
  matrix.values = (T *) malloc(matrix.N_ELEM * sizeof(T));

  for(int i = 0; i < matrix.N_ELEM; i++){
    fscanf(inputFile, "%d %d %lg\n", &matrix.rows[i], &matrix.columns[i], &matrix.values[i]);
    matrix.rows[i]--;
    matrix.columns[i]--;
  }
 
  sortMatrix(matrix); 

  std::cout << matrix << std::endl;

  fclose(inputFile);
  return matrix;
}

}  // namespace Utils
