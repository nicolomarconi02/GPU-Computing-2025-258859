cmake_minimum_required(VERSION 3.16)
project(SpMV)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra")

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(USER_HOME $ENV{HOME})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_definitions("ENABLE_PROFILING=true")

add_subdirectory(${DIR}/external)
add_subdirectory(${DIR}/core)
