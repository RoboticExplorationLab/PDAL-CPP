#include <OsqpEigen/OsqpEigen.h>

#include <iostream>
#include <limits>

#include <Eigen/Dense>

#include "TestDataBenchmark.h"

int main() {
  OsqpEigen::Solver solver;
  pdal::LQProblem problem = getLQProblem();

  const int numDecisionVariables = problem.H.cols();
  const int numEqualityConstraints = problem.G.rows();
  const int numInequalityConstraints = problem.C.rows();
  const int numConstraints = numEqualityConstraints + numInequalityConstraints;

  solver.settings()->setWarmStart(false);
  solver.settings()->setAbsoluteTolerance(1e-6);
  solver.settings()->setRelativeTolerance(1e-6);

  solver.data()->setNumberOfVariables(numDecisionVariables);
  solver.data()->setNumberOfConstraints(numConstraints);
  solver.data()->setHessianMatrix(problem.H);
  solver.data()->setGradient(problem.h);
  pdal::matrix_t stackedConstraintMatrix(numConstraints, numDecisionVariables);
  stackedConstraintMatrix << problem.G.toDense(), problem.C.toDense();
  Eigen::SparseMatrix<double> constraint = stackedConstraintMatrix.sparseView();
  solver.data()->setLinearConstraintsMatrix(constraint);

  pdal::vector_t lowerBound(numConstraints), upperBound(numConstraints);
  lowerBound.head(numEqualityConstraints) = problem.g;
  upperBound.head(numEqualityConstraints) = problem.g;

  // Cx > c
  auto inf = std::numeric_limits<pdal::vector_t::Scalar>::infinity();
  lowerBound.tail(numInequalityConstraints) = problem.c;
  upperBound.tail(numInequalityConstraints).setConstant(inf);

  solver.data()->setLowerBound(lowerBound);
  solver.data()->setUpperBound(upperBound);

  solver.initSolver();

  solver.solveProblem();

  pdal::vector_t sol = solver.getSolution();
  std::cout << sol.transpose() << std::endl;
}
