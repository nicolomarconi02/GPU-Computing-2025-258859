#pragma once

#include <cmath>
#include <profiler/profiler.hpp>
#include "structures/matrix.hpp"

#define SWAP_CONDITION(a,b) (mat.rows[a] < mat.rows[b]) || (mat.rows[a] == mat.rows[b] && mat.columns[a] < mat.columns[b])

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
    if ((mat.rows[j] < pivotRow) ||
        (mat.rows[j] == pivotRow && mat.columns[j] < pivotCol)) {
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

template <typename T>
int findSwapIndex(Matrix<T>& mat, int indexToSwap, int lhs, int rhs) {
  if(lhs >= rhs){
    return lhs;
  }
  int mid = std::trunc((lhs + rhs) / 2);
  if ((mat.rows[mid] < mat.rows[indexToSwap]) ||
      (mat.rows[mid] == mat.rows[indexToSwap] &&
       mat.columns[mid] < mat.columns[indexToSwap])) {
    return findSwapIndex(mat, indexToSwap, mid + 1, rhs);
  }
  else if ((mat.rows[mid] > mat.rows[indexToSwap]) ||
      (mat.rows[mid] == mat.rows[indexToSwap] &&
       mat.columns[mid] > mat.columns[indexToSwap])) {
    return findSwapIndex(mat, indexToSwap, lhs, mid);
  }
  return mid;
}

template <typename T>
int sortMatrixUntil(Matrix<T>& mat, int currentIndex) {
  int swapIndex = findSwapIndex(mat, currentIndex, 0, currentIndex);
  for(int i = swapIndex; i <= currentIndex; i++){
    if(SWAP_CONDITION(currentIndex, i)){
      swap(mat, currentIndex, i); 
    }
  }
  return swapIndex;
}

}  // namespace Utils
