#pragma once

#include "Types.h"

namespace pdal {

struct LQProblem {
  sparseMatrix_t H;
  vector_t h;
  sparseMatrix_t G;
  vector_t g;
  sparseMatrix_t C;
  vector_t c;
};

}  // namespace pdal
