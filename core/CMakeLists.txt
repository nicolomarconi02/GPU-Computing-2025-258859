message("[CORE]")

set(CDIR ${CMAKE_CURRENT_LIST_DIR})

configure_file(${CDIR}/src/defines.hpp.in ${CDIR}/src/defines.hpp)

include_directories(
  SYSTEM
  ${EXTERNAL_DIRS}
  ${CDIR}/src
)

add_library(profiler
  ${CDIR}/src/profiler/profiler.cpp
)

add_library(utils 
  ${CDIR}/src/utils/parser.cpp
)
target_link_libraries(utils PUBLIC mmio profiler)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DIR}/bin/cpu)

add_executable(cpu_csr
  ${CDIR}/executables/cpu_csr/main.cpp
)

target_link_libraries(cpu_csr utils)

if(CUDA_FOUND)
  message(STATUS "FOUND CUDA")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DIR}/bin/gpu)

  add_executable(gpu_csr
    ${CDIR}/executables/gpu_csr/main.cu
  )

  target_include_directories(gpu_csr 
    PRIVATE
    SYSTEM
    ${EXTERNAL_DIRS}
    ${CDIR}/src
  )
  target_link_libraries(gpu_csr utils)

  set_target_properties(gpu_csr PROPERTIES CUDA_ARCHITECTURES "80")
# set_target_properties(gpu_csr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
  
