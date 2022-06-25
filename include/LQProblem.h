#pragma once

#include "Types.h"

namespace pdal {

struct LQProblem {
  sparseMatrix_t H; /** Cost hessian */
  vector_t h;       /** Cost linear term */
  sparseMatrix_t G; /** Linear equality constraints matrix. Gx = g  */
  vector_t g;
  sparseMatrix_t C; /** Linear inequality constraints matrix. Cx >= c  */
  vector_t c;
};

}  // namespace pdal
