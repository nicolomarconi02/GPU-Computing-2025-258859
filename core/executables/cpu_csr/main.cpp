#include "defines.hpp"
#include <cstdint>
#include <iostream>
#include <filesystem>
#include "utils/parser.hpp"
#include "structures/matrix.hpp"
#include "utils/utils.hpp"
#include "operations/cpu_matrix_vec.hpp"
#include "profiler/profiler.hpp"

Mode executionMode = Mode_::CPU;

int main(int argc, char** argv) {
  ScopeProfiler prof("main");
  if (argc != 3) {
    std::cerr << "Usage: ./cpu_csr <select_operation> <path_to_mtx_file>"
              << std::endl;
    exit(1);
  }

  uint8_t operationSelected = std::atoi(argv[1]);
  if (operationSelected >= Operations::MultiplicationTypes::SIZE) {
    std::cerr << "Error uknown operation! Insert:" << std::endl
              << "0 -> sequential multiplication" << std::endl
              << "1 -> parallel multiplication" << std::endl;
    exit(2);
  }

  if (!std::filesystem::is_regular_file(argv[2])) {
    std::cerr << argv[2] << " is not a file" << std::endl;
    exit(3);
  }

  std::cout << "CPU-CSR" << std::endl;

  // parse, store and sorts the matrix of the Matrix Market input file 
  auto retMatrix =
      Utils::parseMatrixMarketFile<indexType_t, dataType_t>(argv[2]);

  if (!retMatrix.has_value()) {
    std::cerr << retMatrix.error() << std::endl;
    exit(4);
  }

  // initialize the vector to be multiplied, set all the values to one for simplicity
  Matrix<indexType_t, dataType_t> vec(MatrixType_::array,
                                      retMatrix.value().N_ELEM);
  for (int i = 0; i < retMatrix.value().N_ELEM; i++) {
    vec.values[i] = 1;
  }

  std::cout << "CSR: " << retMatrix.value().csr[retMatrix.value().N_ROWS]
            << std::endl;

  // initialize the output vector
  Matrix<indexType_t, dataType_t> result(MatrixType_::array,
                                         retMatrix.value().N_ROWS);
  // switch for the operation selected by the user
  switch (operationSelected) {
    case Operations::MultiplicationTypes::Sequential: {
      const indexType_t N_BYTES =
          retMatrix.value().N_ROWS *
              (sizeof(dataType_t) + 2 * sizeof(indexType_t)) +
          retMatrix.value().N_ELEM *
              (sizeof(dataType_t) * 2 + sizeof(indexType_t));
      // start the profiler
      ScopeProfiler prof("multiplication-sequential",
                         2 * retMatrix.value().N_ELEM, N_BYTES);
      auto retMult =
          Operations::sequentialMultiplication(retMatrix.value(), vec);
      if (!retMult.has_value()) {
        std::cerr << retMult.error() << std::endl;
        exit(5);
      }
      result = std::move(retMult.value());
    } break;
    case Operations::MultiplicationTypes::Parallel: {
      const indexType_t N_BYTES =
          retMatrix.value().N_ROWS *
              (sizeof(dataType_t) + 2 * sizeof(indexType_t)) +
          retMatrix.value().N_ELEM *
              (sizeof(dataType_t) * 2 + sizeof(indexType_t));
      // start the profiler
      ScopeProfiler prof("multiplication-parallel",
                         2 * retMatrix.value().N_ELEM, N_BYTES);
      auto retMult = Operations::parallelMultiplication(retMatrix.value(), vec);
      if (!retMult.has_value()) {
        std::cerr << retMult.error() << std::endl;
        exit(5);
      }
      result = std::move(retMult.value());
    } break;
    default:
      std::cerr << "Uknown operation!" << std::endl;
  }

  {
    // measure how much time takes the saving
    ScopeProfiler save("saveResultsToFile");
    Utils::saveResultsToFile(retMatrix.value(), vec, result);
  }
  return 0;
}
