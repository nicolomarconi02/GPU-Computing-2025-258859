#include <cstdint>
#include <cstdio>

namespace Operations {
template <typename T>
__global__ void parallelMultiplication(int N, uint32_t* csr, uint32_t* columns,
                                       T* values, T* vec, T* res) {
  uint32_t count = 0;
  int beginRow = 0;
  int startIndex = threadIdx.x + 1;
  int stride = blockDim.x;
  // printf("GPU\n");
  // printf("csr\n");
  // for(int i = 0; i < N + 1; i++){
  //   printf("%d ", csr[i]);
  // }
  // printf("\ncolumns\n");
  // for(int i = 0; i < N; i++){
  //   printf("%d ", columns[i]);
  // }
  // printf("\nvalues\n");
  // for(int i = 0; i < N; i++){
  //   printf("%f ", values[i]);
  // }
  // printf("\nvec\n");
  // for(int i = 0; i < N; i++){
  //   printf("%f ", vec[i]);
  // }
  // printf("\nres\n");
  // for(int i = 0; i < N; i++){
  //   printf("%f ", res[i]);
  // }
  // printf("\n");
  for (int i = startIndex; i <= N; i += stride) {
    count = csr[i - 1];
    beginRow = count;
    printf("thread: %d, startIndex: %d, i: %d, stride: %d, csr[i-1]: %d\n", threadIdx.x, startIndex, i, stride, csr[i-1]);
    for (; count < beginRow + csr[i] - csr[i - 1]; count++) {
      res[i - 1] += values[count] * vec[columns[count]];
      printf("thread: %d, count: %d, beginRow: %d, i: %d, res[i-1]: %d\n", threadIdx.x, count, beginRow,
             i, res[i - 1]);
    }
  }
}
}  // namespace Operations
