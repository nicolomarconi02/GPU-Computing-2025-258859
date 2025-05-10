cmake_minimum_required(VERSION 3.16)
project(SpMV)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDA)

if(CUDA_FOUND)
  set(COMPILE_GPU_EXEC on)
else(CUDA_FOUND)
  set(COMPILE_GPU_EXEC off)
endif() 
message(STATUS "COMPILE_GPU_EXEC: ${COMPILE_GPU_EXEC}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra")

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(USER_HOME $ENV{HOME})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_definitions("ENABLE_PROFILING=true")

add_subdirectory(${DIR}/external)
add_subdirectory(${DIR}/core)
