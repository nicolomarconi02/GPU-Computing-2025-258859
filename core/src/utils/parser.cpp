#include <cstdint>
#include <cstdio>
#include <iostream>
#include "structures/matrix.hpp"
#include "expected.hpp"
#include "utils/sort_matrix.hpp"

extern "C" {
#include "mmio.h"
}

namespace Utils {
tl::expected<MatrixType, std::string> parseMatrixType(
    const MM_typecode& matcode) {
  MatrixType type = 0;

  if (mm_is_matrix(matcode)) {
    type |= MatrixType_::matrix;
  }
  else{
    return tl::make_unexpected("Uknown typecode 0");
  }

  if(mm_is_coordinate(matcode)){
    type |= MatrixType_::coordinate;
  }
  else if(mm_is_array(matcode)){
    type |= MatrixType_::array;
  }
  else if(mm_is_dense(matcode)){
    type |= MatrixType_::dense;
  }
  else if(mm_is_sparse(matcode)){
    type |= MatrixType_::sparse;
  }
  else{
    return tl::make_unexpected("Uknown typecode 1");
  }

  if(mm_is_complex(matcode)){
    type |= MatrixType_::complex;
  }
  else if(mm_is_real(matcode)){
    type |= MatrixType_::real;
  }
  else if(mm_is_pattern(matcode)){
    type |= MatrixType_::pattern;
  }
  else if(mm_is_integer(matcode)){
    type |= MatrixType_::integer;
  }
  else{
    return tl::make_unexpected("Uknown typecode 2");
  }

  if(mm_is_symmetric(matcode)){
    type |= MatrixType_::symmetric;
  }
  else if(mm_is_general(matcode)){
    type |= MatrixType_::general;
  }
  else if(mm_is_skew(matcode)){
    type |= MatrixType_::skew;
  }
  else if(mm_is_hermitian(matcode)){
    type |= MatrixType_::hermitian;
  }
  else{
    return tl::make_unexpected("Uknown typecode 3");
  }

  return type;
}
}
