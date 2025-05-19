#pragma once

#include <barrier>
#include <cmath>
#include <profiler/profiler.hpp>
#include <thread>
#include "structures/matrix.hpp"

#define SWAP_CONDITION(a, b)     \
  (mat.rows[a] < mat.rows[b]) || \
      (mat.rows[a] == mat.rows[b] && mat.columns[a] < mat.columns[b])

namespace Utils {

//swap utility used in the first version of QuickSort
template <typename indexType, typename dataType>
void swap(Matrix<indexType, dataType>& mat, int i, int j) {
  indexType tmpRow = mat.rows[i];
  indexType tmpCol = mat.columns[i];
  dataType tmpVal = mat.values[i];
  mat.rows[i] = mat.rows[j];
  mat.columns[i] = mat.columns[j];
  mat.values[i] = mat.values[j];
  mat.rows[j] = tmpRow;
  mat.columns[j] = tmpCol;
  mat.values[j] = tmpVal;
}

// partition function for QuickSort
template <typename indexType, typename dataType>
int partition(Matrix<indexType, dataType>& mat, int low, int high) {
  indexType pivotRow = mat.rows[high];
  indexType pivotCol = mat.columns[high];
  int i = (low - 1);
  for (int j = low; j <= high - 1; j++) {
    if ((mat.rows[j] < pivotRow) ||
        (mat.rows[j] == pivotRow && mat.columns[j] < pivotCol)) {
      i++;
      std::swap(mat.rows[i], mat.rows[j]);
      std::swap(mat.cols[i], mat.cols[j]);
      std::swap(mat.values[i], mat.values[j]);
    }
  }
  swap(mat, i + 1, high);
  return (i + 1);
}

// QuickSort not used anymore
template <typename indexType, typename dataType>
void quickSort(Matrix<indexType, dataType>& mat, int low, int high) {
  if (low < high) {
    int pi = partition(mat, low, high);
    quickSort(mat, low, pi - 1);
    quickSort(mat, pi + 1, high);
  }
}

// function call used for the quicksort profiling
template <typename indexType, typename dataType>
void sortMatrix(Matrix<indexType, dataType>& mat) {
  ScopeProfiler prof("quickSort");
  quickSort(mat, 0, mat.N_ELEM - 1);
}

// swap index utility that allows to find the index to swap using binary search
template <typename indexType, typename dataType>
int findSwapIndex(Matrix<indexType, dataType>& mat, int indexToSwap, int lhs,
                  int rhs) {
  if (lhs >= rhs) {
    return lhs;
  }
  int mid = std::trunc((lhs + rhs) / 2);
  if ((mat.rows[mid] < mat.rows[indexToSwap]) ||
      (mat.rows[mid] == mat.rows[indexToSwap] &&
       mat.columns[mid] < mat.columns[indexToSwap])) {
    return findSwapIndex(mat, indexToSwap, mid + 1, rhs);
  } else if ((mat.rows[mid] > mat.rows[indexToSwap]) ||
             (mat.rows[mid] == mat.rows[indexToSwap] &&
              mat.columns[mid] > mat.columns[indexToSwap])) {
    return findSwapIndex(mat, indexToSwap, lhs, mid);
  }
  return mid;
}

// insertion sort like algorithm, not used anymore
template <typename indexType, typename dataType>
int sortMatrixUntil(Matrix<indexType, dataType>& mat, int currentIndex) {
  int swapIndex = findSwapIndex(mat, currentIndex, 0, currentIndex);
  for (int i = swapIndex; i <= currentIndex; i++) {
    if (SWAP_CONDITION(currentIndex, i)) {
      std::swap(mat.rows[i], mat.rows[currentIndex]);
      std::swap(mat.cols[i], mat.cols[currentIndex]);
      std::swap(mat.values[i], mat.values[currentIndex]);
    }
  }
  return swapIndex;
}

// utility for the swap for the parallel bitonic merge sort 
template <typename indexType, typename dataType>
void bitonicCompare(indexType* rows, indexType* cols, dataType* values,
                    indexType i, indexType j, bool ascending) {
  if ((ascending &&
       (rows[i] > rows[j] || (rows[i] == rows[j] && cols[i] > cols[j]))) ||
      (!ascending &&
       (rows[i] < rows[j] || (rows[i] == rows[j] && cols[i] < cols[j])))) {
    std::swap(rows[i], rows[j]);
    std::swap(cols[i], cols[j]);
    std::swap(values[i], values[j]);
  }
}

// Parallel Bitonic Merge Sort
template <typename indexType, typename dataType>
void cpuBitonicSort(Matrix<indexType, dataType>& matrix) {
  indexType size = std::pow(2, std::ceil(std::log2(matrix.N_ELEM)));
  indexType paddingSize = size - matrix.N_ELEM;

  std::vector<indexType> rows(matrix.rows, matrix.rows + matrix.N_ELEM);
  std::vector<indexType> cols(matrix.columns, matrix.columns + matrix.N_ELEM);
  std::vector<dataType> vals(matrix.values, matrix.values + matrix.N_ELEM);

  rows.resize(size, std::numeric_limits<indexType>::max());
  cols.resize(size, std::numeric_limits<indexType>::max());
  vals.resize(size, std::numeric_limits<dataType>::max());

  const int numThreads = std::thread::hardware_concurrency();
  std::barrier sync_point(numThreads);

  for (indexType sequenceSize = 2; sequenceSize <= size; sequenceSize <<= 1) {
    for (indexType distance = sequenceSize >> 1; distance > 0; distance >>= 1) {
      std::vector<std::jthread> threads;
      for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
          for (indexType currentIndex = t; currentIndex < size;
               currentIndex += numThreads) {
            indexType compIndex = currentIndex ^ distance;
            if (compIndex > currentIndex) {
              bool ascending = ((currentIndex & sequenceSize) == 0);
              bitonicCompare(rows.data(), cols.data(), vals.data(),
                             currentIndex, compIndex, ascending);
            }
          }
          sync_point.arrive_and_wait();
        });
      }
      for (auto& t : threads) t.join();
    }
  }

  std::copy(rows.begin(), rows.begin() + matrix.N_ELEM, matrix.rows);
  std::copy(cols.begin(), cols.begin() + matrix.N_ELEM, matrix.columns);
  std::copy(vals.begin(), vals.begin() + matrix.N_ELEM, matrix.values);
}
}  // namespace Utils
