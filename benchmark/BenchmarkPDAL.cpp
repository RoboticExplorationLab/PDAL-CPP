#include "Pdal.h"
#include "TestDataBenchmark.h"

#include <benchmark/benchmark.h>

using namespace pdal;

static void BM_PDAL(benchmark::State& state) {
  Settings settings;
  settings.displayShortSummary = false;
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
