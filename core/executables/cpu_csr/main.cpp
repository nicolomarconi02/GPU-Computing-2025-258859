#include <fstream>
#include <iostream>
#include <filesystem>
#include "utils/parser.hpp"

extern "C"{
  #include "mmio.h"
}

int main(int argc, char** argv){
  if(argc != 2){
    std::cerr << "Usage: ./cpu_csr <path_to_mtx_file>" << std::endl;
    exit(1);
  }

  if(!std::filesystem::is_regular_file(argv[1])){
    std::cerr << argv[1] << " is not a file" << std::endl;
    exit(2);
  }

  auto retMatrix = Utils::parseMatrixMarketFile<double>(argv[1]);

  if(!retMatrix.has_value()){
    std::cerr << retMatrix.error() << std::endl;
    exit(3);
  }

  std::cout << "CPU-CSR" << std::endl;
  return 0;
}
