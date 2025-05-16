#pragma once

#include <defines.hpp>
#include <expected.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "structures/matrix.hpp"
#include "profiler/profiler.hpp"

namespace Utils {

template <typename dataType>
void printVector(const dataType* vec, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << vec[i] << " ";
  }
  std::cout << std::endl;
}

template <typename indexType, typename dataType>
tl::expected<bool, std::string> saveResultsToFile(Matrix<indexType, dataType>& matrix,
                                                  Matrix<indexType, dataType>& vector,
                                                  Matrix<indexType, dataType>& result) {
  if (!std::filesystem::exists(MATRICES_OUTPUT_PATH)) {
    std::filesystem::create_directory(MATRICES_OUTPUT_PATH);
  }
  std::string fileName = "output-" + formatDate(getTimestampMicroseconds()) + "-" +
             formatTime(getTimestampMicroseconds());

  std::ofstream outputFile(MATRICES_OUTPUT_PATH + fileName, std::ofstream::out);
  if (!outputFile.is_open()) {
    return tl::make_unexpected("Error while opening output file");
  }
  outputFile << "INPUT MATRIX" << std::endl << matrix << std::endl;
  outputFile << "_________________________________________________" << std::endl;
  outputFile << "INPUT VECTOR" << std::endl << vector << std::endl;
  outputFile << "_________________________________________________" << std::endl;
  outputFile << "OUTPUT VECTOR" << std::endl << result << std::endl;
  return true;
}
}  // namespace Utils
