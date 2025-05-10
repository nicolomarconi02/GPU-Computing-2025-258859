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
target_link_libraries(utils PRIVATE mmio profiler)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DIR}/bin/cpu)

add_executable(cpu_csr
  ${CDIR}/executables/cpu_csr/main.cpp
)

target_link_libraries(cpu_csr utils profiler)

if(COMPILE_GPU_EXEC)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DIR}/bin/gpu)
  add_executable(gpu_csr
    ${CDIR}/executables/gpu_csr/main.cpp
  )
endif()

