#pragma once

#include <profiler/profiler.hpp>
#include "structures/matrix.hpp"

namespace Utils {

template <typename T>
void swap(Matrix<T>& mat, int i, int j) {
  int tmpRow = mat.rows[i];
  int tmpCol = mat.columns[i];
  T tmpVal = mat.values[i];
  mat.rows[i] = mat.rows[j];
  mat.columns[i] = mat.columns[j];
  mat.values[i] = mat.values[j];
  mat.rows[j] = tmpRow;
  mat.columns[j] = tmpCol;
  mat.values[j] = tmpVal;
}

template <typename T>
int partition(Matrix<T>& mat, int low, int high) {
  uint32_t pivotRow = mat.rows[high];
  uint32_t pivotCol = mat.columns[high];
  int i = (low - 1);
  for (int j = low; j <= high - 1; j++) {
    if ((mat.rows[j] < pivotRow) || (mat.rows[j] == pivotRow && mat.columns[j] < pivotCol)) {
      i++;
      swap(mat, i, j);
    }
  }
  swap(mat, i + 1, high);
  return (i + 1);
}

template <typename T>
void quickSort(Matrix<T>& mat, int low, int high) {
  if (low < high) {
    int pi = partition(mat, low, high);
    quickSort(mat, low, pi - 1);
    quickSort(mat, pi + 1, high);
  }
}

template <typename T>
void sortMatrix(Matrix<T>& mat) {
  ScopeProfiler prof("quickSort");
  quickSort(mat, 0, mat.N_ELEM - 1);
}

}  // namespace Utils
