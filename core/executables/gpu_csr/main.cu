#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <iterator>
#include "utils/parser.hpp"
#include "structures/matrix.hpp"
#include "utils/utils.hpp"
#include "profiler/profiler.hpp"
#include "operations/gpu_matrix_vec.cuh"
#include "utils/sort_matrix_parallel.cuh"
#include "utils/cuda_utils.cuh"

Mode executionMode = Mode_::GPU;

typedef uint32_t indexType_t;
typedef double dataType_t;

int main(int argc, char **argv) {
  ScopeProfiler prof("main");
  if (argc != 3) {
    std::cerr << "Usage: ./gpu_csr <select_operation> <path_to_mtx_file>"
              << std::endl;
    exit(1);
  }

  uint8_t operationSelected = std::atoi(argv[1]);
  if (operationSelected >= Operations::MultiplicationTypes::SIZE) {
    std::cerr << "Error uknown operation! Insert:" << std::endl
              << "0 -> thread per row multiplication" << std::endl
              << "1 -> element wise multiplication" << std::endl
              << "2 -> warp multiplication" << std::endl;
    exit(2);
  }

  if (!std::filesystem::is_regular_file(argv[2])) {
    std::cerr << argv[2] << " is not a file" << std::endl;
    exit(3);
  }

  std::cout << "GPU-CSR" << std::endl;

  auto retMatrix =
      Utils::parseMatrixMarketFile<indexType_t, dataType_t>(argv[2]);

  if (!retMatrix.has_value()) {
    std::cerr << retMatrix.error() << std::endl;
    exit(4);
  }

  Utils::parallelSort(retMatrix.value());
  retMatrix.value().computeCSR();

  Matrix<indexType_t, dataType_t> matrix = std::move(retMatrix.value());

  std::cout << "matrix csr: " << matrix.csr[matrix.N_ROWS] << std::endl;

  Matrix<indexType_t, dataType_t> vec(MatrixType_::array, matrix.N_ELEM);
  for (int i = 0; i < matrix.N_ELEM; i++) {
    vec.values[i] = 1;
  }

  Matrix<indexType_t, dataType_t> resMat(MatrixType_::array, matrix.N_ROWS);

  indexType_t *csr, *columns;
  dataType_t *values, *array, *res1, *res2;
  // GPU allocation
  CUDA_CHECK(cudaMalloc(&csr, (matrix.N_ROWS + 1) * sizeof(indexType_t)));
  CUDA_CHECK(cudaMalloc(&columns, matrix.N_ELEM * sizeof(indexType_t)));
  CUDA_CHECK(cudaMalloc(&values, matrix.N_ELEM * sizeof(dataType_t)));
  CUDA_CHECK(cudaMalloc(&array, matrix.N_ELEM * sizeof(dataType_t)));

  res1 = (dataType_t *)calloc(matrix.N_ROWS, sizeof(dataType_t));
  if (!res1) {
    std::cerr << "Calloc error on res1!" << std::endl;
    return EXIT_FAILURE;
  }

  CUDA_CHECK(cudaMalloc(&res2, matrix.N_ROWS * sizeof(dataType_t)));

  // GPU copy
  CUDA_CHECK(cudaMemcpy(csr, matrix.csr,
                        (matrix.N_ROWS + 1) * sizeof(indexType_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(columns, matrix.columns,
                        matrix.N_ELEM * sizeof(indexType_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(values, matrix.values,
                        matrix.N_ELEM * sizeof(dataType_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(array, vec.values, matrix.N_ELEM * sizeof(dataType_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(res2, res1, matrix.N_ROWS * sizeof(dataType_t),
                        cudaMemcpyHostToDevice));

  std::cout << "Completed all the CUDA malloc and memcpy correctly!"
            << std::endl;
  switch (operationSelected) {
    case Operations::MultiplicationTypes::ThreadPerRow: {
      const indexType_t N_BLOCKS = COMPUTE_N_BLOCKS(indexType_t, matrix.N_ROWS);
      const indexType_t N_THREAD = COMPUTE_N_THREAD(indexType_t, matrix.N_ROWS);
      const indexType_t N_BYTES =
          matrix.N_ELEM * (sizeof(dataType_t) * 2 + sizeof(indexType_t)) +
          matrix.N_ROWS * (sizeof(dataType_t) + 2 * sizeof(indexType_t));
      ScopeProfiler pMult("multiplication-thread-per-row", 2 * matrix.N_ELEM, N_BYTES);
      Operations::parallelMultiplicationThreadPerRow<<<N_BLOCKS, N_THREAD>>>(
          (indexType_t)matrix.N_ROWS, csr, columns, values, array, res2);
      cudaDeviceSynchronize();
    } break;
    case Operations::MultiplicationTypes::ElementWise: {
      const indexType_t N_BLOCKS = COMPUTE_N_BLOCKS(indexType_t, matrix.N_ROWS);
      const indexType_t N_THREAD = COMPUTE_N_THREAD(indexType_t, matrix.N_ROWS);
      const indexType_t N_BYTES =
          matrix.N_ELEM * (sizeof(dataType_t) * 3 + 2 * sizeof(indexType_t));
      ScopeProfiler pMult("multiplication-element-wise", 2 * matrix.N_ELEM, N_BYTES);
      Operations::parallelMultiplicationElementWise<<<N_BLOCKS, N_THREAD>>>(
          (indexType_t)matrix.N_ROWS, csr, columns, values, array, res2);
      cudaDeviceSynchronize();
    } break;
    case Operations::MultiplicationTypes::Warp: {
      const indexType_t N_WARPS = 4;
      const indexType_t N_THREAD = N_WARPS * 32;
      const indexType_t N_BLOCKS = (matrix.N_ROWS + N_WARPS - 1) / N_WARPS;
      const indexType_t N_BYTES =
          matrix.N_ELEM * (sizeof(dataType_t) * 2 + sizeof(indexType_t)) +
          matrix.N_ROWS * (sizeof(dataType_t) + 2 * sizeof(indexType_t));
      ScopeProfiler pMult("multiplication-warp", 2 * matrix.N_ELEM, N_BYTES);
      Operations::parallelMultiplicationWarp<<<N_BLOCKS, N_THREAD>>>(
          (indexType_t)matrix.N_ROWS, csr, columns, values, array, res2);
      cudaDeviceSynchronize();
    } break;
    default:
      std::cerr << "Uknown operation!" << std::endl;
  }

  cudaMemcpy(resMat.values, res2, (matrix.N_ROWS) * sizeof(dataType_t),
             cudaMemcpyDeviceToHost);

  std::cout << "save: " << std::endl;
  Utils::saveResultsToFile(matrix, vec, resMat);
  CUDA_CHECK(cudaFree(csr));
  CUDA_CHECK(cudaFree(columns));
  CUDA_CHECK(cudaFree(values));
  CUDA_CHECK(cudaFree(array));
  CUDA_CHECK(cudaFree(res2));
  free(res1);
  return 0;
}
