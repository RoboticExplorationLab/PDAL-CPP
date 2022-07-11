#include <OsqpEigen/OsqpEigen.h>

#include <Eigen/Dense>
#include <limits>

#include <benchmark/benchmark.h>

#include "TestDataBenchmark.h"

static void BM_OSQP(benchmark::State& state) {
  OsqpEigen::Solver solver;
  pdal::LQProblem problem = getLQProblem();

  const int numDecisionVariables = problem.H.cols();
  const int numEqualityConstraints = problem.G.rows();
  const int numInequalityConstraints = problem.C.rows();
  const int numConstraints = numEqualityConstraints + numInequalityConstraints;

  solver.settings()->setWarmStart(false);
  solver.settings()->setVerbosity(false);

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

  for (auto _ : state) {
    solver.solveProblem();
  }
}

BENCHMARK(BM_OSQP)->Unit(benchmark::kMillisecond);
