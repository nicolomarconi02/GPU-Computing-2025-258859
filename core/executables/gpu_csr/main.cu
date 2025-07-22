#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <defines.hpp>
#include <iostream>
#include <filesystem>
#include <iterator>
#include <nvToolsExt.h>
#include <utils/sort_matrix.hpp>
#include "utils/parser.hpp"
#include "structures/matrix.hpp"
#include "utils/utils.hpp"
#include "profiler/profiler.hpp"
#include "operations/gpu_matrix_vec.cuh"
#include "utils/sort_matrix_parallel.cuh"
#include "utils/cuda_utils.cuh"

Mode executionMode = Mode_::GPU;

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
              << "2 -> warp multiplication" << std::endl
              << "3 -> warp multiplication loop" << std::endl
              << "4 -> warp multiplication tiled" << std::endl
              << "5 -> CuSparse" << std::endl;
    exit(2);
  }

  if (!std::filesystem::is_regular_file(argv[2])) {
    std::cerr << argv[2] << " is not a file" << std::endl;
    exit(3);
  }

  std::cout << "GPU-CSR" << std::endl;

  Profiler::getProfiler().setMatrixFileName(argv[2]);
  // parse and store matrix market file
  auto retMatrix =
      Utils::parseMatrixMarketFile<indexType_t, dataType_t>(argv[2]);

  std::cout << "parsed matrix market file" << std::endl;
  if (!retMatrix.has_value()) {
    std::cerr << retMatrix.error() << std::endl;
    exit(4);
  }

  // sort with parallel bitonic sort and compute csr on the sorted matrix
  Utils::parallelSort(retMatrix.value());

  std::cout << "completed parallel sort" << std::endl;
  retMatrix.value().computeCSR();
  std::cout << "completed csr computation" << std::endl;

  Matrix<indexType_t, dataType_t> matrix = std::move(retMatrix.value());

  // initialize dense vector for the multiplication (all values set to one for
  // simplicity)
  Matrix<indexType_t, dataType_t> vec(MatrixType_::array, matrix.N_ELEM);
  for (int i = 0; i < matrix.N_ELEM; i++) {
    vec.values[i] = 1;
  }

  // initialize result vector
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

  std::cout << "Completed all the CUDA malloc and memcpy correctly!"
            << std::endl;
  // run warmup and measure cycles all together
  for (int run = 0; run < GPU_N_WARMUP_RUNS + GPU_N_MEASURE_RUNS; run++) {
    CUDA_CHECK(cudaMemcpy(res2, res1, matrix.N_ROWS * sizeof(dataType_t),
                          cudaMemcpyHostToDevice));
    // switch for the selected operation by the user
    switch (operationSelected) {
      case Operations::MultiplicationTypes::ThreadPerRow: {
        const indexType_t N_BLOCKS =
            COMPUTE_N_BLOCKS(indexType_t, matrix.N_ROWS);
        const indexType_t N_THREAD =
            COMPUTE_N_THREAD(indexType_t, matrix.N_ROWS);
        const indexType_t N_BYTES =
            matrix.N_ELEM * (sizeof(dataType_t) * 2 + sizeof(indexType_t)) +
            matrix.N_ROWS * (sizeof(dataType_t) + 2 * sizeof(indexType_t));
        // profile the multiplication only after the warmup cycles
        if (run >= GPU_N_WARMUP_RUNS) {
          ScopeProfiler pMult("multiplication-thread-per-row",
                              2 * matrix.N_ELEM, N_BYTES);
          Operations::
              parallelMultiplicationThreadPerRow<<<N_BLOCKS, N_THREAD>>>(
                  (indexType_t)matrix.N_ROWS, csr, columns, values, array,
                  res2);
          cudaDeviceSynchronize();
        } else {
          Operations::
              parallelMultiplicationThreadPerRow<<<N_BLOCKS, N_THREAD>>>(
                  (indexType_t)matrix.N_ROWS, csr, columns, values, array,
                  res2);
          cudaDeviceSynchronize();
        }
      } break;
      case Operations::MultiplicationTypes::ElementWise: {
        const indexType_t N_BLOCKS =
            COMPUTE_N_BLOCKS(indexType_t, matrix.N_ROWS);
        const indexType_t N_THREAD =
            COMPUTE_N_THREAD(indexType_t, matrix.N_ROWS);
        const indexType_t N_BYTES =
            matrix.N_ELEM * (sizeof(dataType_t) * 3 + 2 * sizeof(indexType_t));
        // profile the multiplication only after the warmup cycles
        if (run >= GPU_N_WARMUP_RUNS) {
          ScopeProfiler pMult("multiplication-element-wise", 2 * matrix.N_ELEM,
                              N_BYTES);
          Operations::parallelMultiplicationElementWise<<<N_BLOCKS, N_THREAD>>>(
              (indexType_t)matrix.N_ROWS, csr, columns, values, array, res2);
          cudaDeviceSynchronize();
        } else {
          Operations::parallelMultiplicationElementWise<<<N_BLOCKS, N_THREAD>>>(
              (indexType_t)matrix.N_ROWS, csr, columns, values, array, res2);
          cudaDeviceSynchronize();
        }
      } break;
      case Operations::MultiplicationTypes::Warp: {
        const indexType_t N_WARPS = 4;
        const indexType_t N_THREAD = N_WARPS * 32;
        const indexType_t N_BLOCKS = (matrix.N_ROWS + N_WARPS - 1) / N_WARPS;
        const indexType_t N_BYTES =
            matrix.N_ELEM * (sizeof(dataType_t) * 2 + sizeof(indexType_t)) +
            matrix.N_ROWS * (sizeof(dataType_t) + 2 * sizeof(indexType_t));
        // profile the multiplication only after the warmup cycles
        if (run >= GPU_N_WARMUP_RUNS) {
          ScopeProfiler pMult("multiplication-warp", 2 * matrix.N_ELEM,
                              N_BYTES);
          nvtxRangePush("multiplication-warp");
          Operations::parallelMultiplicationWarp<<<N_BLOCKS, N_THREAD>>>(
              (indexType_t)matrix.N_ROWS, csr, columns, values, array, res2);
          cudaDeviceSynchronize();
          nvtxRangePop();
        } else {
          Operations::parallelMultiplicationWarp<<<N_BLOCKS, N_THREAD>>>(
              (indexType_t)matrix.N_ROWS, csr, columns, values, array, res2);
          cudaDeviceSynchronize();
        }
      } break;
      case Operations::MultiplicationTypes::WarpLoop: {
        const indexType_t N_BLOCKS =
            COMPUTE_N_BLOCKS(indexType_t, matrix.N_ROWS);
        const indexType_t N_THREAD =
            COMPUTE_N_THREAD(indexType_t, matrix.N_ROWS);
        const indexType_t N_BYTES =
            matrix.N_ELEM * (sizeof(dataType_t) * 2 + sizeof(indexType_t)) +
            matrix.N_ROWS * (sizeof(dataType_t) + 2 * sizeof(indexType_t));
        // profile the multiplication only after the warmup cycles
        if (run >= GPU_N_WARMUP_RUNS) {
          ScopeProfiler pMult("multiplication-warp-loop", 2 * matrix.N_ELEM,
                              N_BYTES);
          Operations::parallelMultiplicationWarpLoop<<<N_BLOCKS, N_THREAD>>>(
              (indexType_t)matrix.N_ROWS, csr, columns, values, array, res2);
          cudaDeviceSynchronize();
        } else {
          Operations::parallelMultiplicationWarpLoop<<<N_BLOCKS, N_THREAD>>>(
              (indexType_t)matrix.N_ROWS, csr, columns, values, array, res2);
          cudaDeviceSynchronize();
        }
      } break;
      case Operations::MultiplicationTypes::WarpTiled: {
        const indexType_t N_THREAD = 512;
        const indexType_t ROWS_PER_BLOCK = N_THREAD / 32;
        const indexType_t BLOCK_COL_CHUNK = 2024;

        const indexType_t N_BLOCKS =
            (matrix.N_ROWS + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        const indexType_t N_BYTES =
            matrix.N_ELEM * (sizeof(dataType_t) * 2 + sizeof(indexType_t)) +
            matrix.N_ROWS * (sizeof(dataType_t) + 2 * sizeof(indexType_t));
        std::vector<indexType_t> colStart, colEnd, rowBlock;
        Utils::computeBlockColRangesOptimized(
            (indexType_t)matrix.N_ROWS, matrix.csr, matrix.columns,
            ROWS_PER_BLOCK, BLOCK_COL_CHUNK, colStart, colEnd, rowBlock);
        for (int b = 0; b < min(N_BLOCKS, 10); b++) {
          std::cout << "Block " << b
                    << " span = " << (colEnd[b] - colStart[b] + 1)
                    << " (start=" << colStart[b] << ", end=" << colEnd[b]
                    << ")\n";
        }

        const indexType_t N_TILES = colStart.size();

        indexType_t *d_colStart, *d_colEnd, *d_rowBlock;
        CUDA_CHECK(cudaMalloc(&d_colStart, (N_TILES) * sizeof(indexType_t)));
        CUDA_CHECK(cudaMalloc(&d_colEnd, (N_TILES) * sizeof(indexType_t)));
        CUDA_CHECK(cudaMalloc(&d_rowBlock, (N_TILES) * sizeof(indexType_t)));

        CUDA_CHECK(cudaMemcpy(d_colStart, colStart.data(),
                              N_TILES * sizeof(indexType_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colEnd, colEnd.data(),
                              N_TILES * sizeof(indexType_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rowBlock, rowBlock.data(),
                              N_TILES * sizeof(indexType_t),
                              cudaMemcpyHostToDevice));

        std::cout << "Completed all the CUDA malloc for warptiled and memcpy "
                     "correctly!"
                  << std::endl;
        // profile the multiplication only after the warmup cycles
        if (run >= GPU_N_WARMUP_RUNS) {
          ScopeProfiler pMult("multiplication-warp-tiled", 2 * matrix.N_ELEM,
                              N_BYTES);
          nvtxRangePush("multiplication-warp-tiled");
          Operations::parallelMultiplicationWarpTiled<
              indexType_t, dataType_t, ROWS_PER_BLOCK, BLOCK_COL_CHUNK>
              <<<N_TILES, N_THREAD, BLOCK_COL_CHUNK * sizeof(dataType_t)>>>(
                  (indexType_t)matrix.N_ROWS, csr, columns, values, array, res2,
                  d_colStart, d_colEnd, d_rowBlock);
          cudaDeviceSynchronize();
          nvtxRangePop();
        } else {
          Operations::parallelMultiplicationWarpTiled<
              indexType_t, dataType_t, ROWS_PER_BLOCK, BLOCK_COL_CHUNK>
              <<<N_TILES, N_THREAD, BLOCK_COL_CHUNK * sizeof(dataType_t)>>>(
                  (indexType_t)matrix.N_ROWS, csr, columns, values, array, res2,
                  d_colStart, d_colEnd, d_rowBlock);
          cudaDeviceSynchronize();
        }
        CUDA_CHECK(cudaFree(d_colStart));
        CUDA_CHECK(cudaFree(d_colEnd));
        CUDA_CHECK(cudaFree(d_rowBlock));
      } break;
      case Operations::MultiplicationTypes::CuSparse: {
        // profile the multiplication only after the warmup cycles
        if (run >= GPU_N_WARMUP_RUNS) {
          ScopeProfiler pMult("multiplication-CuSparse", 2 * matrix.N_ELEM, 0);
          Operations::SpMVcuSparse(
              (indexType_t)matrix.N_ROWS, (indexType_t)matrix.N_COLS,
              (indexType_t)matrix.N_ELEM, csr, columns, values, array, res2);
        } else {
          Operations::SpMVcuSparse(
              (indexType_t)matrix.N_ROWS, (indexType_t)matrix.N_COLS,
              (indexType_t)matrix.N_ELEM, csr, columns, values, array, res2);
        }
      } break;
      default:
        std::cerr << "Uknown operation!" << std::endl;
    }
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
