#pragma once

#include <vector>

#include <SparseLDL/CodeGen/SparseLDLGenerated.h>
#include <qdldl.h>

#include "LQProblem.h"
#include "Settings.h"
#include "Types.h"

namespace pdal {
class PdalSolver {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using settings_t = pdal::Settings;

  PdalSolver(settings_t settings);

  /**
   * Setup the linear-quadratic problem. Cost, equality and inequality constraints will be copied to an internal buffer.
   * It assumes the sparse pattern of all input matrices are fixed and the a tree structure(elimination tree)
   * representing those sparsities will be generated after calling this function.
   *
   * @param[in] lqProblem: Linear-quadratic problem data
   * @return isSucceed
   */
  bool setupProblem(const LQProblem& lqProblem);

  Eigen::PermutationMatrix<Eigen::Dynamic>& getPermutations() { return perm_; }

  /**
   * Solve the buffered LQ-problem. The initial value is required and result will be calculated in place.
   *
   * @param[in, out] x: Initial value before function call. Result after function call
   * @return isSucceed
   */
  bool solve(vector_t& x);

  /**
   * Newton step. Inner loop of the augmented lagrangian.
   *
   * @param lambda
   * @param mu
   * @param rho
   * @param y
   * @param w
   * @param x
   * @return int: Number of iteration. -1 if it reached the max iteration.
   */
  int newtonSolve(const vector_t& lambda, const vector_t& mu, const PDAL_float_t rho, vector_t& y, vector_t& w,
                  vector_t& x);

  /**
   * Evaluate equality, inequality constraints and the selection matrix Ic.
   *
   * @param mu
   * @param x
   */
  void evaluateConstraints(const vector_t& mu, const vector_t& x);

  /**
   * Evaluate the primal and dual residuals.
   *
   * @param lambda
   * @param mu
   * @param x
   */
  void evaluatePrimalDualResidual(const vector_t& lambda, const vector_t& mu, const vector_t& x);

  const std::vector<QDLDL_int>& etree() const { return etree_; }
  const settings_t& settings() const { return settings_; }
  PDAL_int_t numDecisionVariables() const { return numDecisionVariables_; }

 private:
  void resize();
  const settings_t settings_;

  LQProblem lqProblem_;

  PDAL_int_t numDecisionVariables_{};
  PDAL_int_t numEqConstraints_{};
  PDAL_int_t numIneqConstraints_{};

  vector_t eqConstraints_{};   /** Evaluation of the equality constraints */
  vector_t ineqConstraints_{}; /** Evaluation of the inequality constraints */
  sparseMatrix_t Ic_{};

  vector_t dualResidual_;
  vector_t primalResidual_;

  std::vector<QDLDL_int> Lnz_;
  std::vector<QDLDL_int> etree_;
  QDLDL_int sumLnz_;

  Eigen::PermutationMatrix<Eigen::Dynamic> perm_;
  DxCollection<PDAL_float_t, 8, 4> Dx_;
  DxCollection<PDAL_float_t, 8, 4> DxInv_;
  LxCollection<PDAL_float_t, 8, 4> Lx_;
};
}  // namespace pdal