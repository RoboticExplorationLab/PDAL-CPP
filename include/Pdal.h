#pragma once

#include <vector>

#include <qdldl.h>

#include "LQProblem.h"
#include "Settings.h"
#include "Types.h"

namespace pdal {
class PdalSolver {
 public:
  using settings_t = pdal::Settings;

  PdalSolver() = default;
  PdalSolver(settings_t settings);

  bool setupProblem(const LQProblem& ldProblem);
  // bool solve(vector_t& x);

  const std::vector<QDLDL_int>& etree() { return etree_; }

 private:
  void resize();
  const settings_t settings_;

  LQProblem lqProblem_;

  PDAL_int_t numDecisionVariables_;
  PDAL_int_t numEqConstraints_;
  PDAL_int_t numIneqConstraints_;

  vector_t lambda_; /** Equality constraints multiplier */
  vector_t mu_;     /** Inequality constraints multiplier */

  PDAL_float_t rho_; /** Penalty */

  vector_t y_; /** Dual variable equality constraints */
  vector_t w_; /** Dual variable inequality constraints */

  vector_t eqConstraints_;   /** Evaluation of the equality constraints */
  vector_t ineqConstraints_; /** Evaluation of the inequality constraints */

  std::vector<QDLDL_int> Lnz_;
  std::vector<QDLDL_int> etree_;
  QDLDL_int sumLnz_;
};
}  // namespace pdal