#include "Pdal.h"
#include "TestDataBenchmark.h"

#include <benchmark/benchmark.h>

#include "GenerateRandomProblem.h"
#include "HelperFunctions.h"

using namespace pdal;

Eigen::VectorXi indexFromLength(int from, int length) {
  Eigen::VectorXi ind(length);
  for (int i = 0; i < length; ++i) ind(i) = from + i;
  return ind;
}

static void BM_PDAL(benchmark::State& state) {
  Settings settings;
  settings.displayShortSummary = false;
  settings.displayRunTime = false;
  settings.primalResidualTolerance = 1e-6;
  settings.dualResidualTolerance = 1e-6;
  // settings.innerTolerance = 1e-1;

  PdalSolver solver(settings);

  solver.setupProblem(getLQProblem());

  vector_t x(solver.numDecisionVariables());

  for (auto _ : state) {
    state.PauseTiming();
    x.setZero();
    state.ResumeTiming();

    solver.solve(x);
  }
}

static void BM_PDAL_DoubleInegratorIn4D(benchmark::State& state) {
  constexpr static const int Dim = 4;  // Double integrator dimension
  constexpr static const int nx = Dim * 2;
  constexpr static const int nu = Dim;
  constexpr static const int N = LxCollection<PDAL_float_t, nx, nu>::traits::numStages;  // numStages
  constexpr static const int numDecisionVariables = N * (nx + nu);
  constexpr static const int numConstraints = N * nx;

  Settings settings;
  settings.displayRunTime = false;
  settings.displayShortSummary = false;
  settings.primalResidualTolerance = 1e-6;
  settings.dualResidualTolerance = 1e-6;
  PdalSolver solver(settings);
  LQProblem lqProblem;

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

  solver.setupProblem(lqProblem);
  auto& perm = solver.getPermutations();
  perm.resize(numDecisionVariables + numConstraints);
  for (int i = 0; i < N; ++i) {
    perm.indices().segment((nu + nx + nx) * i, nu) = indexFromLength((nu + nx) * i, nu);
    perm.indices().segment((nu + nx + nx) * i + nu, nx) = indexFromLength(numDecisionVariables + i * nx, nx);
    perm.indices().segment((nu + nx + nx) * i + nu + nx, nx) = indexFromLength((nu + nx) * i + nu, nx);
  }

  vector_t x = vector_t::Zero(solver.numDecisionVariables());
  for (auto _ : state) {
    state.PauseTiming();
    x.setZero();
    state.ResumeTiming();

    solver.solve(x);
  }
}

// BENCHMARK(BM_PDAL)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_PDAL_DoubleInegratorIn4D)->Unit(benchmark::kMillisecond);
