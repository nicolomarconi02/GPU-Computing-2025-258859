#include <cstdint>
#include <iostream>
#include <filesystem>
#include "utils/parser.hpp"
#include "structures/matrix.hpp"
#include "utils/utils.hpp"
#include "operations/cpu_matrix_vec.hpp"
#include "profiler/profiler.hpp"

typedef uint32_t indexType_t;
typedef double dataType_t;

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
              << "0 -> sequential multiplication" << std::endl;
    exit(2);
  }

  if (!std::filesystem::is_regular_file(argv[2])) {
    std::cerr << argv[2] << " is not a file" << std::endl;
    exit(3);
  }

  std::cout << "CPU-CSR" << std::endl;

  auto retMatrix =
      Utils::parseMatrixMarketFile<indexType_t, dataType_t>(argv[2]);

  if (!retMatrix.has_value()) {
    std::cerr << retMatrix.error() << std::endl;
    exit(4);
  }

  Matrix<indexType_t, dataType_t> vec(MatrixType_::array,
                                      retMatrix.value().N_ELEM);
  for (int i = 0; i < retMatrix.value().N_ELEM; i++) {
    vec.values[i] = 1;
  }

  std::cout << "CSR: " << retMatrix.value().csr[retMatrix.value().N_ROWS]
            << std::endl;
  Matrix<indexType_t, dataType_t> result(MatrixType_::array,
                                         retMatrix.value().N_ROWS);
  switch (operationSelected) {
    case Operations::MultiplicationTypes::Sequential: {
      const indexType_t N_BYTES = retMatrix.value().N_ROWS * (sizeof(dataType_t) + 2 * sizeof(indexType_t)) + retMatrix.value().N_ELEM * (sizeof(dataType_t) * 2 + sizeof(indexType_t)); 
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
    default:
      std::cerr << "Uknown operation!" << std::endl;
  }

  Utils::saveResultsToFile(retMatrix.value(), vec, result);
  return 0;
}
