#pragma once

#include <iostream>
namespace Utils{
  template <typename T> 
  void printVector(const T* vec, int size){
    for(int i = 0; i < size; i++){
      std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
  }
}
