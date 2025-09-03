# GPU-Computing-2025-258859
This repository contains the C++/CUDA project of the GPU-Computing course.

## Objective
Write a program that computes a **sparse matrix-dense vector** multiplication (SpMV). Insert both CPU and GPU implementation with `CUDA`. 

## Project structure
```
├── core
│   ├── CMakeLists.txt
│   ├── executables
│   │   ├── cpu_csr
│   │   │   └── main.cpp
│   │   └── gpu_csr
│   │       └── main.cu
│   └── src
│       ├── defines.hpp
│       ├── defines.hpp.in
│       ├── operations
│       │   ├── cpu_matrix_vec.hpp
│       │   └── gpu_matrix_vec.cuh
│       ├── profiler
│       │   ├── profiler.cpp
│       │   └── profiler.hpp
│       ├── structures
│       │   └── matrix.hpp
│       └── utils
│           ├── cuda_utils.cuh
│           ├── parser.cpp
│           ├── parser.hpp
│           ├── sort_matrix.hpp
│           ├── sort_matrix_parallel.cuh
│           └── utils.hpp
├── external
│   ├── CMakeLists.txt
│   ├── expected
│   │   ├── cmake
│   │   │   └── tl-expected-config.cmake.in
│   │   ├── CMakeLists.txt
│   │   ├── COPYING
│   │   ├── expected.natvis
│   │   ├── include
│   │   │   └── tl
│   │   │       └── expected.hpp
│   │   ├── README.md
│   │   └── tests
│   │       ├── assertions.cpp
│   │       ├── assignment.cpp
│   │       ├── bases.cpp
│   │       ├── constexpr.cpp
│   │       ├── constructors.cpp
│   │       ├── emplace.cpp
│   │       ├── extensions.cpp
│   │       ├── issues.cpp
│   │       ├── main.cpp
│   │       ├── noexcept.cpp
│   │       ├── observers.cpp
│   │       ├── relops.cpp
│   │       ├── swap.cpp
│   │       └── test.cpp
│   └── mmio
│       ├── example_read.c
│       ├── example_write
│       ├── example_write.c
│       ├── libmmio.so
│       ├── Makefile
│       ├── mmio.c
│       ├── mmio.h
│       ├── mmio.o
│       └── README.md
├── Makefile
├── README.md
├── report
│   └── GPU_Computing_SpMV.pdf
└── run-job.sh
```
The ``core`` directory contains all the executables and source files of the project.
The ``external`` directory keeps all the submodule used in the project:
 - expected: library for handling return values of functions
 - mmio: library for the **Matrix Market files** parsing
   
The ``report`` directory contains the report pdf file

## Installation
Clone this repository, enter the cloned directory and then run: 
```
git submodule update --recursive --init
```

## Building
> **_WARNING:_**  In order to build this project it is necessary to have CMake installed on the system.

In the root of the project run:
```
make build
```
With this command the compilation of the project starts. If `CUDA` is installed and found on the system, `CMake` will automatically compile also the CUDA files and generate the GPU executable.

The compilation results will be generated in the `bin` folder:
```
bin
└── cpu
|   └── cpu_csr
└── gpu
    └── gpu_csr
```

## Usage
To run the program, choose the executable to use and run:
```
./bin/cpu/cpu_csr <select_multiplication_to_use> <path_to_mtx_file>

./bin/gpu/gpu_csr <select_multiplication_to_use> <path_to_mtx_file>
```

For the CPU executables the type of multiplication to use are two:
```
0 -> sequential multiplication
1 -> parallel multiplication
```

For the GPU executables the multiplications are three:
```
0 -> thread per row multiplication
1 -> element wise multiplication
2 -> warp multiplication
3 -> warp multiplication loop
4 -> warp multiplication tiled
5 -> merge based multiplication
6 -> merge based multiplication v4
7 -> CuSparse
```

## Outputs
When the program is executed there will be generated two output directories:
```
├── output_matrices
│   └── output-<date>-<time>
└── profiler_session
    └── session-<date>-<time>
```
The `output_matrices` directory keeps all the information about the input matrix parsed, sorted and shows the CSR computed, but also the dense-vector (which for simplicity is always composed by all ones, but it could be also generated randomically), and the result vector of the matrix vector multiplication.

The `profiler_session` directory stores the informations about the execution time of each block of code and also its throughput and bandwidth.
