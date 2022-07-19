#include <OsqpEigen/OsqpEigen.h>

#include <iostream>
#include <limits>

#include <Eigen/Dense>

#include "TestDataBenchmark.h"

#include <SparseLDL/CodeGen/SparseLDLGenerated.h>

#include <GenerateRandomProblem.h>
#include <HelperFunctions.h>

using namespace pdal;

void setupProblem(LQProblem& lqProblem) {
  constexpr static const int Dim = 4;  // Double integrator dimension
  constexpr static const int nx = Dim * 2;
  constexpr static const int nu = Dim;
  constexpr static const int N = LxCollection<PDAL_float_t, nx, nu>::traits::numStages;  // numStages
  constexpr static const int numDecisionVariables = N * (nx + nu);
  constexpr static const int numConstraints = N * nx;

  vector_s_t<PDAL_float_t> x0(Dim * 2);
  x0.setZero();
  x0.head<Dim>() << 3, 4, 5, 6, -2, 1, 6, 7;

  for (int i = 0; i < N; i++) {
    lqProblem.dynamics.push_back(getDoubleIntegratorDynamicsInDimN<PDAL_float_t, Dim>());
    lqProblem.cost.push_back(getDoubleIntegratorCostInDimN<PDAL_float_t, Dim>());
  }
  lqProblem.cost.push_back(getDoubleIntegratorCostInDimN<PDAL_float_t, Dim>());
  lqProblem.H = getCostMatrix(lqProblem.cost, numDecisionVariables).sparseView();
  lqProblem.G = getConstraintMatrix(lqProblem.dynamics, numConstraints, numDecisionVariables).sparseView();
  lqProblem.h = vector_s_t<PDAL_float_t>::Zero(lqProblem.H.rows());
  lqProblem.g = vector_s_t<PDAL_float_t>::Zero(lqProblem.G.rows());
  lqProblem.g.head<Dim * 2>() = -lqProblem.dynamics[0].A * x0;
  lqProblem.C = sparseMatrix_t();
  lqProblem.c = vector_s_t<PDAL_float_t>();
}

int main() {
  OsqpEigen::Solver solver;
  LQProblem problem;
  setupProblem(problem);

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
  matrix_t stackedConstraintMatrix(numConstraints, numDecisionVariables);
  stackedConstraintMatrix << problem.G.toDense(), problem.C.toDense();
  Eigen::SparseMatrix<PDAL_float_t> constraint = stackedConstraintMatrix.sparseView();
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
  // std::cout << sol.transpose() << std::endl;
}
