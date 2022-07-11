#include "Pdal.h"
#include "TestDataBenchmark.h"

using namespace pdal;

int main() {
  Settings settings;
  settings.displayShortSummary = false;
  settings.displayRunTime = true;

  // settings.primalResidualTolerance = 1e-1;
  // settings.dualResidualTolerance = 1e-1;
  // settings.innerTolerance = 1e-1;

  PdalSolver solver(settings);

  solver.setupProblem(getLQProblem());

  vector_t x(solver.numDecisionVariables());

  solver.solve(x);
}
