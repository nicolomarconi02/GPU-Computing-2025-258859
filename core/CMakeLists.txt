message("[CORE]")

set(CDIR ${CMAKE_CURRENT_LIST_DIR})

configure_file(${CDIR}/src/defines.hpp.in ${CDIR}/src/defines.hpp)

include_directories(
  ${CDIR}/src
)
include_directories(
  SYSTEM
  ${EXTERNAL_DIRS}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DIR}/bin/cpu)

add_executable(cpu-csr
  ${CDIR}/executables/cpu-csr/main.cpp
)

target_link_libraries(cpu-csr )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DIR}/bin/gpu)

add_executable(gpu-csr
  ${CDIR}/executables/gpu-csr/main.cpp
)

