#include <cstdint>
#include <iostream>
#include <filesystem>
#include "utils/parser.hpp"
#include "structures/matrix.hpp"
#include "utils/utils.hpp"
#include "operations/cpu_matrix_vec.hpp"
#include "profiler/profiler.hpp"

typedef double data_t;

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

  auto retMatrix = Utils::parseMatrixMarketFile<data_t>(argv[1]);

  if(!retMatrix.has_value()){
    std::cerr << retMatrix.error() << std::endl;
    exit(3);
  }

  data_t* vec = (data_t*) malloc(sizeof(data_t) * retMatrix.value().N_ELEM);
  for(int i = 0; i < retMatrix.value().N_ELEM; i++){
    vec[i] = 1;
  }

  std::cout << retMatrix.value() << std::endl;
  
  std::cout << "start vec" << std::endl;
  Utils::printVector(vec, retMatrix.value().N_ELEM);

  auto retMult = Operations::multiplication(retMatrix.value(), vec);
  
  std::cout << "ret vec" << std::endl;
  Utils::printVector(retMult.value(), retMatrix.value().N_ROWS);

  std::cout << "CPU-CSR" << std::endl;
  retMatrix->freeMatrix();
  free(vec);
  free(retMult.value());
  return 0;
}
