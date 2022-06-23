#include "Pdal.h"

namespace pdal {

bool PdalSolver::solve(const sparseMatrix_t& H, const vector_t& h, const sparseMatrix_t& G, const vector_t& g,
                       const sparseMatrix_t& C, const vector_t& c, vector_t& x) {
  PDAL_int_t numDecisionVariables = h.rows();
  if (x.size() != numDecisionVariables) {
    throw std::invalid_argument("The size of the decision variabels x is different from the # of cols of H");
  }
  PDAL_int_t numEqConstraints = G.rows();
  PDAL_int_t numIneqConstraints = C.rows();

  vector_t lambda = vector_t::Zero(numEqConstraints);
  vector_t mu = vector_t::Zero(numIneqConstraints);

  PDAL_float_t rho = settings_.initialRho;

  vector_t y = vector_t::Zero(numEqConstraints);
  vector_t w = vector_t::Zero(numIneqConstraints);

  vector_t eqConstraints = -g;
  eqConstraints.noalias() += G * x;

  vector_t ineqConstraints = c;
  ineqConstraints.noalias() -= C * x;

  boolVector_t Ic(numIneqConstraints);

  for (PDAL_int_t outerIterNum = 0; outerIterNum < settings_.maxOuterIter; ++outerIterNum) {
    for (PDAL_int_t n = 0; n < numIneqConstraints; ++n) {
      if (ineqConstraints(n) > 0 || mu(n) > 0) {
        Ic(n) = 1;
      } else {
        ineqConstraints(n) = 0;
      }
    }

    y = lambda;
    y.noalias() += rho * eqConstraints;

    w = mu;
    w.noalias() += rho * ineqConstraints;

    // inner loop
    for (PDAL_int_t innerIterNum = 0; innerIterNum < settings_.maxInnerIter; ++innerIterNum) {
    }
  }
}
}  // namespace pdal