#include "Pdal.h"
#include "TestDataBenchmark.h"

#include <benchmark/benchmark.h>

using namespace pdal;

static void BM_PDAL(benchmark::State& state) {
  Settings settings;
  settings.displayShortSummary = false;
  settings.displayRunTime = false;
  // settings.primalResidualTolerance = 1e-1;
  // settings.dualResidualTolerance = 1e-1;
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

BENCHMARK(BM_PDAL)->Unit(benchmark::kMillisecond);
