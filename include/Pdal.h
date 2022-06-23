/**
 * @file PDAL.h
 * @author your name (you@domain.com)
 * @brief
 *
 */

#pragma once

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

  PdalSolver();
  bool solve(const sparseMatrix_t& H, const vector_t& h, const sparseMatrix_t& G, const vector_t& g,
             const sparseMatrix_t& C, const vector_t& c, vector_t& x);

 private:
  const settings_t settings_;
};

}  // namespace pdal