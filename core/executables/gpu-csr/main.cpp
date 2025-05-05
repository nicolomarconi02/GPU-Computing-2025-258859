#include <iostream>

extern "C"{
  #include "mmio.h"
}

int main(int argc, char** argv){
  std::cout << "GPU-CSR" << std::endl;
  return 0;
}
