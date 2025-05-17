cmake_minimum_required(VERSION 3.16)
project(SpMV LANGUAGES C CXX)

find_package(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(CUDA_FOUND)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fopenmp -ffast-math -g -Wall -Wextra")

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(USER_HOME $ENV{HOME})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(${DIR}/external)
add_subdirectory(${DIR}/core)
