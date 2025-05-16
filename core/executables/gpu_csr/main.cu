#include <algorithm>
#include <cstdint>
#include <iostream>
#include <filesystem>
#include "utils/parser.hpp"
#include "structures/matrix.hpp"
#include "utils/utils.hpp"
#include "operations/cpu_matrix_vec.hpp"
#include "profiler/profiler.hpp"
#include "operations/gpu_matrix_vec.cuh"
#include "utils/sort_matrix_parallel.cuh"

Mode executionMode = Mode_::GPU;

#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << #call << ": "                           \
                << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

typedef double dtype_t;

int main(int argc, char **argv) {
  ScopeProfiler prof("main");
  if (argc != 2) {
    std::cerr << "Usage: ./gpu_csr <path_to_mtx_file>" << std::endl;
    exit(1);
  }

  if (!std::filesystem::is_regular_file(argv[1])) {
    std::cerr << argv[1] << " is not a file" << std::endl;
    exit(2);
  }

  std::cout << "GPU-CSR" << std::endl;

  auto retMatrix = Utils::parseMatrixMarketFile<dtype_t>(argv[1]);

  if (!retMatrix.has_value()) {
    std::cerr << retMatrix.error() << std::endl;
    exit(3);
  }

  std::cout << "retmat csr: " << std::endl;
  for (int i = 0; i < retMatrix.value().N_ROWS + 1; i++) {
    std::cout << retMatrix.value().csr[i] << " ";
  }

  Matrix<dtype_t> matrix = std::move(retMatrix.value());

  std::cout << "matrix csr: " << std::endl;
  for (int i = 0; i < matrix.N_ROWS + 1; i++) {
    std::cout << matrix.csr[i] << " ";
  }

  Matrix<dtype_t> vec(MatrixType_::array, matrix.N_ELEM);
  for (int i = 0; i < matrix.N_ELEM; i++) {
    vec.values[i] = 1;
  }

  std::cout << matrix << std::endl;

  std::cout << "start vec" << std::endl;
  std::cout << vec;

  Matrix<dtype_t> resMat(MatrixType_::array, matrix.N_ROWS);
  const uint8_t N_THREAD = 16;
  const uint8_t N_BLOCKS = 1;

  uint32_t *csr, *columns;
  dtype_t *values, *array, *res1, *res2;
  // GPU allocation
  CUDA_CHECK(cudaMalloc(&csr, (matrix.N_ROWS + 1) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&columns, matrix.N_ELEM * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&values, matrix.N_ELEM * sizeof(dtype_t)));
  CUDA_CHECK(cudaMalloc(&array, matrix.N_ELEM * sizeof(dtype_t)));

  res1 = (dtype_t *)calloc(matrix.N_ROWS, sizeof(dtype_t));
  if (!res1) {
    std::cerr << "Calloc error on res1!" << std::endl;
    return EXIT_FAILURE;
  }

  CUDA_CHECK(cudaMalloc(&res2, matrix.N_ROWS * sizeof(dtype_t)));

  // GPU copy
  CUDA_CHECK(cudaMemcpy(csr, matrix.csr, (matrix.N_ROWS + 1) * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(columns, matrix.columns,
                        matrix.N_ELEM * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(values, matrix.values, matrix.N_ELEM * sizeof(dtype_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(array, vec.values, matrix.N_ELEM * sizeof(dtype_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(res2, res1, matrix.N_ROWS * sizeof(dtype_t),
                        cudaMemcpyHostToDevice));

  std::cout << "Completed all the CUDA malloc and memcpy correctly!"
            << std::endl;
  {
    ScopeProfiler("multiplication");
    Operations::parallelMultiplication<<<N_BLOCKS, N_THREAD>>>(
        matrix.N_ROWS, csr, columns, values, array, res2);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(resMat.values, res2, (matrix.N_ROWS) * sizeof(dtype_t),
             cudaMemcpyDeviceToHost);

  std::cout << "res: " << std::endl;
  for (int i = 0; i < matrix.N_ROWS; i++) {
    std::cout << resMat.values[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "save: " << std::endl;
  Utils::saveResultsToFile(matrix, vec, resMat);
  cudaFree(csr);
  cudaFree(columns);
  cudaFree(values);
  cudaFree(array);
  free(res1);
  cudaFree(res2);
  return 0;
}
