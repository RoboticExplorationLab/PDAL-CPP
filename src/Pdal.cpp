/**
 * @file Pdal.cpp
 * @author Fu Zhengyu (zhengfuaj@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-24
 *
 */

#include "Pdal.h"

#include <iostream>
#include <sstream>
#include <vector>

namespace pdal {

namespace {
void emplaceBackMatrixToTriplet(const PDAL_int_t startRow, const PDAL_int_t startCol, const sparseMatrix_t& mat,
                                std::vector<triplet_t>& triplet, const PDAL_int_t rowBound = -1,
                                const PDAL_int_t colBound = -1) {
  for (PDAL_int_t j = 0; j < mat.outerSize(); ++j) {
    for (sparseMatrix_t::InnerIterator it(mat, j); it; ++it) {
      assert(rowBound == -1 || startRow + it.row() <= rowBound);
      assert(colBound == -1 || startCol + it.col() <= colBound);

      triplet.emplace_back(startRow + it.row(), startCol + it.col(), it.value());
    }
  }
};

void emplaceRepeatDiagonalToTriplet(const PDAL_int_t startRow, const PDAL_float_t value, const PDAL_int_t n,
                                    std::vector<triplet_t>& triplet, const PDAL_int_t rowBound = -1,
                                    const PDAL_int_t colBound = -1) {
  assert(rowBound == -1 || startRow + n - 1 <= rowBound);
  assert(colBound == -1 || startRow + n - 1 <= colBound);

  for (PDAL_int_t j = 0; j < n; ++j) {
    triplet.emplace_back(startRow + j, startRow + j, value);
  }
};
}  // namespace

PdalSolver::PdalSolver(settings_t settings) : settings_(settings){};

bool PdalSolver::setupProblem(const LQProblem& ldProblem) {
  lqProblem_ = ldProblem;

  resize();

  // Construct elimination tree
  PDAL_int_t Hn = numDecisionVariables_ + numEqConstraints_ + numIneqConstraints_;
  PDAL_int_t nnzHessian = (Hn + 1) * Hn / 2;
  std::vector<triplet_t> hessianTriplets;
  hessianTriplets.reserve(nnzHessian);
  emplaceBackMatrixToTriplet(0, 0, lqProblem_.H.triangularView<Eigen::Upper>(), hessianTriplets);
  emplaceBackMatrixToTriplet(0, numDecisionVariables_, lqProblem_.G.transpose(), hessianTriplets);
  emplaceBackMatrixToTriplet(0, numDecisionVariables_ + numEqConstraints_, lqProblem_.C.transpose(), hessianTriplets);
  emplaceRepeatDiagonalToTriplet(numDecisionVariables_, 1.0, numEqConstraints_ + numIneqConstraints_, hessianTriplets);

  sparseMatrix_t hessian(Hn, Hn);
  hessian.setFromTriplets(hessianTriplets.cbegin(), hessianTriplets.cend());
  assert(hessian.isCompressed());
  assert(hessian.nonZeros() <= nnzHessian);

  Lnz_.resize(Hn);
  etree_.resize(Hn);
  std::vector<PDAL_int_t> flag(Hn);
  sumLnz_ = QDLDL_etree(Hn, hessian.outerIndexPtr(), hessian.innerIndexPtr(), flag.data(), Lnz_.data(), etree_.data());
  if (sumLnz_ < 0) {
    throw std::runtime_error("QDLDL_etree failed.");
  }

  return true;
}

void PdalSolver::resize() {
  assert(lqProblem_.H.rows() == lqProblem_.h.rows() || (lqProblem_.H.rows() == 0 || lqProblem_.h.rows() == 0));
  assert(lqProblem_.G.rows() == lqProblem_.g.rows());
  assert(lqProblem_.C.rows() == lqProblem_.c.rows());

  numDecisionVariables_ = lqProblem_.H.rows();
  numEqConstraints_ = lqProblem_.G.rows();
  numIneqConstraints_ = lqProblem_.C.rows();

  eqConstraints_.resize(numEqConstraints_);
  ineqConstraints_.resize(numIneqConstraints_);

  primalResidual_.resize(numDecisionVariables_);
  dualResidual_.resize(numEqConstraints_ + numIneqConstraints_);
}

void PdalSolver::evaluateConstraints(const vector_t& mu, const vector_t& x) {
  // G* x - g
  eqConstraints_ = -lqProblem_.g;
  eqConstraints_.noalias() += lqProblem_.G * x;

  // -C*x + c
  ineqConstraints_ = lqProblem_.c;
  ineqConstraints_.noalias() -= lqProblem_.C * x;

  std::vector<triplet_t> IcTriplets;
  IcTriplets.reserve(numIneqConstraints_);
  for (PDAL_int_t n = 0; n < numIneqConstraints_; ++n) {
    if (ineqConstraints_(n) > 0 || mu(n) > 0) {
      IcTriplets.emplace_back(n, n, 1.0);
    }
  }
  Ic_.resize(numIneqConstraints_, numIneqConstraints_);
  Ic_.setFromTriplets(IcTriplets.cbegin(), IcTriplets.cend());
  assert(Ic_.nonZeros() <= numIneqConstraints_);
}

void PdalSolver::evaluatePrimalDualResidual(const vector_t& lambda, const vector_t& mu, const vector_t& x) {
  // primalResidual = H * xResult + h + G '*lambda - C' * mu;
  primalResidual_ = lqProblem_.h;
  primalResidual_.noalias() += lqProblem_.H * x;
  primalResidual_.noalias() += lqProblem_.G.transpose() * lambda;
  primalResidual_.noalias() -= lqProblem_.C.transpose() * mu;

  dualResidual_.head(numEqConstraints_) = eqConstraints_;
  dualResidual_.tail(numIneqConstraints_) = ineqConstraints_.cwiseMax(0);
}

bool PdalSolver::solve(vector_t& x) {
  if (x.size() != numDecisionVariables_) {
    std::stringstream ss;
    ss << "The size of the decision variables x is " << x.size()
       << " which is different from the size of decision variables (" << numDecisionVariables_
       << "). Run setupProblem first or check the input size.";
    throw std::invalid_argument(ss.str());
  }

  PDAL_float_t rho = settings_.initialRho;
  vector_t lambda = vector_t::Zero(numEqConstraints_); /** Equality constraints multiplier */
  vector_t mu = vector_t::Zero(numIneqConstraints_);   /** Inequality constraints multiplier */

  for (PDAL_int_t outerIterNum = 0; outerIterNum < settings_.maxOuterIter; ++outerIterNum) {
    evaluateConstraints(mu, x);
    evaluatePrimalDualResidual(lambda, mu, x);

    PDAL_float_t dualResidualNorm = dualResidual_.norm();
    PDAL_float_t primalResidualNorm = primalResidual_.norm();

    int innerItr;
    if (settings_.displayShortSummary) {
      if (outerIterNum == 0) {
        std::cerr << "Initial norm(dualResidual): " << dualResidualNorm
                  << " norm(primalResidual): " << primalResidualNorm << "\n";
      } else {
        std::cerr << "Iter: " << outerIterNum << " InnerItr: " << innerItr
                  << " norm(dualResidual): " << dualResidualNorm << " norm(primalResidual): " << primalResidualNorm
                  << "\n";
      }
    }
    if (dualResidualNorm < settings_.dualResidualTolerance && primalResidualNorm < settings_.primalResidualTolerance) {
      return true;
    }

    vector_t y = lambda + rho * eqConstraints_;
    vector_t w = mu + rho * Ic_ * ineqConstraints_;
    innerItr = newtonSolve(lambda, mu, rho, y, w, x);

    // lambda = lambda + rho * (G*xResult-g);
    lambda += rho * (lqProblem_.G * x);
    lambda -= rho * lqProblem_.g;

    // mu = mu - rho * (C * x - c);
    mu -= rho * (lqProblem_.C * x);
    mu += rho * lqProblem_.c;
    mu = mu.cwiseMax(0);

    rho *= settings_.amplificationRho;
  }

  return false;
}

int PdalSolver::newtonSolve(const vector_t& lambda, const vector_t& mu, const PDAL_float_t rho, vector_t& y,
                            vector_t& w, vector_t& x) {
  for (PDAL_int_t innerIterNum = 0; innerIterNum < settings_.maxInnerIter; ++innerIterNum) {
    // L = H * x + h + G ' * y - C'*Ic'*w;
    vector_t L = lqProblem_.h;
    L.noalias() += lqProblem_.H * x;
    L.noalias() += lqProblem_.G.transpose() * y;
    vector_t tmp = Ic_ * w;
    L.noalias() -= lqProblem_.C.transpose() * tmp;

    PDAL_int_t Hn = numDecisionVariables_ + numEqConstraints_ + numIneqConstraints_;
    vector_t residual(Hn);
    residual << L, eqConstraints_ + 1 / rho * (lambda - y), Ic_ * ineqConstraints_ + 1 / rho * (mu - w);
    if (residual.norm() < settings_.innerTolerance) {
      return innerIterNum;
    }

    PDAL_int_t nnzHessian = ((Hn + 1) * Hn) / 2;
    std::vector<triplet_t> hessianTriplets;
    hessianTriplets.reserve(nnzHessian);
    emplaceBackMatrixToTriplet(0, 0, lqProblem_.H.triangularView<Eigen::Upper>(), hessianTriplets);
    emplaceBackMatrixToTriplet(0, numDecisionVariables_, lqProblem_.G.transpose(), hessianTriplets);
    emplaceBackMatrixToTriplet(0, numDecisionVariables_ + numEqConstraints_, (-Ic_ * lqProblem_.C).transpose(),
                               hessianTriplets);
    emplaceRepeatDiagonalToTriplet(numDecisionVariables_, -1.0 / rho, numEqConstraints_ + numIneqConstraints_,
                                   hessianTriplets);

    sparseMatrix_t hessian(Hn, Hn);
    hessian.setFromTriplets(hessianTriplets.cbegin(), hessianTriplets.cend());
    assert(hessian.nonZeros() <= nnzHessian);

    std::vector<QDLDL_int> Lp(Hn + 1), Li(sumLnz_);
    std::vector<QDLDL_float> Lx(sumLnz_);
    std::vector<QDLDL_float> D(Hn), Dinv(Hn);
    std::vector<QDLDL_bool> bwork(Hn);
    std::vector<QDLDL_int> iwork(3 * Hn);
    std::vector<QDLDL_float> fwork(Hn);

    QDLDL_int positiveValuesInD = QDLDL_factor(Hn, hessian.outerIndexPtr(), hessian.innerIndexPtr(), hessian.valuePtr(),
                                               Lp.data(), Li.data(), Lx.data(), D.data(), Dinv.data(), Lnz_.data(),
                                               etree_.data(), bwork.data(), iwork.data(), fwork.data());
    if (positiveValuesInD < 0) {
      throw std::runtime_error("LDLT factorazation fail.");
    }

    vector_t dz = -residual;
    QDLDL_solve(Hn, Lp.data(), Li.data(), Lx.data(), Dinv.data(), dz.data());
    x += dz.head(numDecisionVariables_);
    y += dz.segment(numDecisionVariables_, numEqConstraints_);
    w += dz.tail(numIneqConstraints_);

    evaluateConstraints(mu, x);
  }
  return -1;
}

}  // namespace pdal