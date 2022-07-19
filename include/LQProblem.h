#pragma once

#include "SparseLDL/Types.h"
#include "Types.h"

namespace pdal {

constexpr int Nx = 8;
constexpr int Nu = 4;

struct LQProblem {
  sparseMatrix_t H; /** Cost hessian */
  vector_t h;       /** Cost linear term */
  sparseMatrix_t G; /** Linear equality constraints matrix. Gx = g  */
  vector_t g;
  sparseMatrix_t C; /** Linear inequality constraints matrix. Cx >= c  */
  vector_t c;

  DynamicsAlignedStdVector<PDAL_float_t, Nx, Nu> dynamics;
  CostAlignedStdVector<PDAL_float_t, Nx, Nu> cost;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace pdal
