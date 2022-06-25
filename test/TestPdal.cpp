#include "Pdal.h"

#include <gtest/gtest.h>

#include "TestData.h"

using namespace pdal;

TEST(Pdal, elimination_tree_construction) {
  Eigen::Map<const sparseMatrix_t> H(An, An, Ap[An], Ap, Ai, Ax);
  PdalSolver solver;
  LQProblem lqProblem;

  lqProblem.H = H;
  lqProblem.h = vector_t();

  lqProblem.G = sparseMatrix_t();
  lqProblem.g = vector_t();

  lqProblem.C = sparseMatrix_t();
  lqProblem.c = vector_t();

  solver.setupProblem(lqProblem);
  std::vector<QDLDL_int> work(An), Lnz(An), etree(An);
  QDLDL_int sumLnz = QDLDL_etree(An, Ap, Ai, work.data(), Lnz.data(), etree.data());
  EXPECT_GE(sumLnz, 0);
  EXPECT_EQ(solver.etree(), etree);
}

TEST(Pdal, test) {
  PdalSolver solver;
  LQProblem lqProblem;

  lqProblem.H = get_H();
  lqProblem.h = get_h();

  lqProblem.G = get_G();
  lqProblem.g = get_g();

  lqProblem.C = get_C();
  lqProblem.c = get_c();

  solver.setupProblem(lqProblem);

  vector_t x = vector_t::Zero(solver.numDecisionVariables());
  solver.solve(x);

  EXPECT_LE((lqProblem.G * x - lqProblem.g).norm(), solver.settings().dualResidualTolerance);  // Equality constraints

  vector_t inEq = lqProblem.C * x - lqProblem.c;
  EXPECT_LE(inEq.cwiseMin(0).norm(), solver.settings().dualResidualTolerance)
      << "Ineq: " << inEq.transpose();  // Inequality constraints
}