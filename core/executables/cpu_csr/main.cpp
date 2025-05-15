#include <cstdint>
#include <iostream>
#include <filesystem>
#include "utils/parser.hpp"
#include "structures/matrix.hpp"
#include "utils/utils.hpp"
#include "operations/cpu_matrix_vec.hpp"
#include "profiler/profiler.hpp"

typedef double dtype_t;

int main(int argc, char** argv){
  ScopeProfiler prof("main");
  if(argc != 2){
    std::cerr << "Usage: ./cpu_csr <path_to_mtx_file>" << std::endl;
    exit(1);
  }

  if(!std::filesystem::is_regular_file(argv[1])){
    std::cerr << argv[1] << " is not a file" << std::endl;
    exit(2);
  }

  std::cout << "CPU-CSR" << std::endl;

  auto retMatrix = Utils::parseMatrixMarketFile<dtype_t>(argv[1]);

  if(!retMatrix.has_value()){
    std::cerr << retMatrix.error() << std::endl;
    exit(3);
  }

  Matrix<dtype_t> vec(MatrixType_::array, retMatrix.value().N_ELEM);
  for(int i = 0; i < retMatrix.value().N_ELEM; i++){
    vec.values[i] = 1;
  }

  std::cout << retMatrix.value() << std::endl;
  
  std::cout << "start vec" << std::endl;
  std::cout << vec;

  auto retMult = Operations::sequentialMultiplication(retMatrix.value(), vec);
  if(!retMult.has_value()){
    std::cerr << retMult.error() << std::endl;
    exit(4);
  }
  
  std::cout << "ret vec" << std::endl;
  std::cout << retMult.value();
  Utils::saveResultsToFile(retMatrix.value(), vec, retMult.value());
  return 0;
}
