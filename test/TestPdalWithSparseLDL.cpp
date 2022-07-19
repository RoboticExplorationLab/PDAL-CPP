#include "Pdal.h"
#include "SparseLDL/CodeGen/SparseLDLGenerated.h"

#include <gtest/gtest.h>

#include "GenerateRandomProblem.h"
#include "HelperFunctions.h"
#include "SparseLDL/Types.h"

using namespace pdal;
using namespace std;

Eigen::VectorXi indexFromLength(int from, int length) {
  Eigen::VectorXi ind(length);
  for (int i = 0; i < length; ++i) ind(i) = from + i;
  return ind;
}

TEST(Pdal, test) {
  constexpr static const int Dim = 4;  // Double integrator dimension
  constexpr static const int nx = Dim * 2;
  constexpr static const int nu = Dim;
  constexpr static const int N = LxCollection<double, nx, nu>::traits::numStages;  // numStages
  constexpr static const int numDecisionVariables = N * (nx + nu);
  constexpr static const int numConstraints = N * nx;

  Settings settings;
  PdalSolver solver(settings);
  LQProblem lqProblem;

  vector_s_t<double> x0(Dim * 2);
  x0.setZero();
  x0.head<Dim>() << 3, 4, 5, 6, -2, 1, 6, 7;

  for (int i = 0; i < N; i++) {
    lqProblem.dynamics.push_back(getDoubleIntegratorDynamicsInDimN<double, Dim>());
    lqProblem.cost.push_back(getDoubleIntegratorCostInDimN<double, Dim>());
  }
  lqProblem.cost.push_back(getDoubleIntegratorCostInDimN<double, Dim>());
  lqProblem.H = getCostMatrix(lqProblem.cost, numDecisionVariables).sparseView();
  lqProblem.G = getConstraintMatrix(lqProblem.dynamics, numConstraints, numDecisionVariables).sparseView();
  lqProblem.h = vector_s_t<double>::Zero(lqProblem.H.rows());
  lqProblem.g = vector_s_t<double>::Zero(lqProblem.G.rows());
  lqProblem.g.head<Dim * 2>() = -lqProblem.dynamics[0].A * x0;
  lqProblem.C = sparseMatrix_t();
  lqProblem.c = vector_s_t<double>();

  cout << lqProblem.c.size() << " " << lqProblem.c.rows() << endl;

  solver.setupProblem(lqProblem);
  auto& perm = solver.getPermutations();
  perm.resize(numDecisionVariables + numConstraints);
  for (int i = 0; i < N; ++i) {
    perm.indices().segment((nu + nx + nx) * i, nu) = indexFromLength((nu + nx) * i, nu);
    perm.indices().segment((nu + nx + nx) * i + nu, nx) = indexFromLength(numDecisionVariables + i * nx, nx);
    perm.indices().segment((nu + nx + nx) * i + nu + nx, nx) = indexFromLength((nu + nx) * i + nu, nx);
  }

  vector_t x = vector_t::Zero(solver.numDecisionVariables());
  solver.solve(x);

  EXPECT_LE((lqProblem.G * x - lqProblem.g).norm(), solver.settings().dualResidualTolerance);  // Equality constraint
}