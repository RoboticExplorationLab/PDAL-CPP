#pragma once

#include <vector>

#include <qdldl.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "Settings.h"
#include "Types.h"

namespace pdal {
class PdalSolver {
 public:
  using settings_t = pdal::Settings;

  using sparseMatrix_t = Eigen::SparseMatrix<PDAL_float_t>;
  using vector_t = Eigen::Matrix<PDAL_float_t, Eigen::Dynamic, 1>;
  using boolVector_t = Eigen::Matrix<PDAL_bool_t, Eigen::Dynamic, 1>;
  using triplet_t = Eigen::Triplet<PDAL_float_t>;

  PdalSolver() = default;
  PdalSolver(settings_t settings);

  bool setupProblem(const sparseMatrix_t& H, const vector_t& h, const sparseMatrix_t& G, const vector_t& g,
                    const sparseMatrix_t& C, const vector_t& c);
  // bool solve(vector_t& x);

  const std::vector<QDLDL_int>& etree() { return etree_; }

 private:
  void resize();
  const settings_t settings_;

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

  const sparseMatrix_t* H_ptr_;
  const sparseMatrix_t* G_ptr_;
  const sparseMatrix_t* C_ptr_;

  const vector_t* h_ptr_;
  const vector_t* g_ptr_;
  const vector_t* c_ptr_;

  std::vector<QDLDL_int> Lnz_;
  std::vector<QDLDL_int> etree_;
  QDLDL_int sumLnz_;
};
}  // namespace pdal