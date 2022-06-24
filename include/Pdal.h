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
  bool solve(vector_t& x);
  int newtonSolve(const vector_t& lambda, const vector_t& mu, const PDAL_float_t rho, vector_t& y, vector_t& w,
                  vector_t& x);
  void evaluateConstraints(const vector_t& mu, const vector_t& x);
  void evaluatePrimalDualResidual(const vector_t& lambda, const vector_t& mu, const vector_t& x);

  const std::vector<QDLDL_int>& etree() const { return etree_; }
  const settings_t& settings() const { return settings_; }
  PDAL_int_t numDecisionVariables() const { return numDecisionVariables_; }

 private:
  void resize();
  const settings_t settings_;

  LQProblem lqProblem_;

  PDAL_int_t numDecisionVariables_;
  PDAL_int_t numEqConstraints_;
  PDAL_int_t numIneqConstraints_;

  vector_t eqConstraints_;   /** Evaluation of the equality constraints */
  vector_t ineqConstraints_; /** Evaluation of the inequality constraints */
  sparseMatrix_t Ic_;

  vector_t primalResidual_;
  vector_t dualResidual_;

  std::vector<QDLDL_int> Lnz_;
  std::vector<QDLDL_int> etree_;
  QDLDL_int sumLnz_;
};
}  // namespace pdal