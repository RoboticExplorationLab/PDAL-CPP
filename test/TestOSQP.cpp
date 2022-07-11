#include <OsqpEigen/OsqpEigen.h>

#include <iostream>
#include <limits>

#include <Eigen/Dense>

#include "LQProblem.h"
#include "TestData.h"

int main() {
  auto checkStatus = [](bool success) {
    if (!success) {
      std::cerr << "\033[1;31mFailed\033[0m" << std::endl;
    }
  };
  OsqpEigen::Solver solver;
  pdal::LQProblem problem;

  problem.H = get_H();
  problem.h = get_h();

  problem.G = get_G();
  problem.g = get_g();

  problem.C = get_C();
  problem.c = get_c();

  const int numDecisionVariables = problem.H.cols();
  const int numEqualityConstraints = problem.G.cols();
  const int numInequalityConstraints = problem.C.cols();
  const int numConstraints = numEqualityConstraints + numInequalityConstraints;

  solver.settings()->setWarmStart(false);

  solver.data()->setNumberOfVariables(numDecisionVariables);
  solver.data()->setNumberOfConstraints(numConstraints);
  checkStatus(solver.data()->setHessianMatrix(problem.H));
  checkStatus(solver.data()->setGradient(problem.h));

  pdal::matrix_t stackedConstraintMatrix(numConstraints, numDecisionVariables);
  stackedConstraintMatrix << problem.G.toDense(), problem.C.toDense();
  Eigen::SparseMatrix<double> constraint = stackedConstraintMatrix.sparseView();
  checkStatus(solver.data()->setLinearConstraintsMatrix(constraint));

  pdal::vector_t lowerBound(numConstraints), upperBound(numConstraints);

  auto inf = std::numeric_limits<pdal::vector_t::Scalar>::infinity();
  lowerBound.head(numEqualityConstraints) = problem.g;
  upperBound.head(numEqualityConstraints) = problem.g;
  // Cx > c
  lowerBound.tail(numInequalityConstraints).setConstant(-inf);
  upperBound.tail(numInequalityConstraints).setConstant(inf);

  checkStatus(solver.data()->setLowerBound(lowerBound));
  checkStatus(solver.data()->setUpperBound(upperBound));

  checkStatus(solver.initSolver());

  solver.solveProblem();

  pdal::vector_t sol = solver.getSolution();
  std::cout << sol.transpose() << std::endl;

  std::cout << lowerBound.transpose() << std::endl;
}
