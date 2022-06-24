/**
 * @file Pdal.cpp
 * @author Fu Zhengyu (zhengfuaj@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-24
 *
 */

#include "Pdal.h"

#include <vector>

namespace pdal {

namespace {
void emplaceBackMatrixToTriplet(const PDAL_int_t startRow, const PDAL_int_t startCol, const sparseMatrix_t& mat,
                                std::vector<triplet_t>& triplet) {
  for (PDAL_int_t j = 0; j < mat.outerSize(); ++j) {
    for (sparseMatrix_t::InnerIterator it(mat, j); it; ++it) {
      triplet.emplace_back(startRow + it.row(), startCol + it.col(), it.value());
    }
  }
};

void emplaceRepeatDiagonalToTriplet(const PDAL_int_t startRow, const PDAL_float_t value, const PDAL_int_t n,
                                    std::vector<triplet_t>& triplet) {
  for (PDAL_int_t j = 0; j < n; ++j) {
    triplet.emplace_back(startRow + n, startRow + n, value);
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

  lambda_.resize(numEqConstraints_);
  mu_.resize(numIneqConstraints_);

  y_.resize(numEqConstraints_);
  w_.resize(numIneqConstraints_);

  eqConstraints_.resize(numEqConstraints_);
  ineqConstraints_.resize(numIneqConstraints_);
}

// bool PdalSolver::solve(vector_t& x) {
//   if (x.size() != numDecisionVariables_) {
//     throw std::invalid_argument("The size of the decision variables x is different from the # of cols of H");
//   }

//   PDAL_float_t rho_ = settings_.initialRho;

//   vector_t eqConstraints_ = -g;
//   eqConstraints_.noalias() += G * x;

//   vector_t ineqConstraints_ = c;
//   ineqConstraints_.noalias() -= C * x;

//   std::vector<triplet_t> IcTriplets;
//   IcTriplets.reserve(numIneqConstraints_);
//   sparseMatrix_t Ic(numIneqConstraints_, numIneqConstraints_);

//   for (PDAL_int_t outerIterNum = 0; outerIterNum < settings_.maxOuterIter; ++outerIterNum) {
//     for (PDAL_int_t n = 0; n < numIneqConstraints_; ++n) {
//       if (ineqConstraints_(n) > 0 || mu_(n) > 0) {
//         IcTriplets.emplace_back(n, n, 1.0);
//       }
//     }
//     Ic.setFromTriplets(IcTriplets.cbegin(), IcTriplets.cend());

//     // y_ = lambda_ + rho_ * eqConstraints_;
//     y_ = lambda_;
//     y_.noalias() += rho_ * eqConstraints_;

//     // w_ = mu_ + rho_ * Ic * ineqConstraints_;
//     w_ = mu_;
//     w_.noalias() += rho_ * Ic * ineqConstraints_;
//   }
// }

// bool PdalSolver::innerLoop(const sparseMatrix_t& H, const vector_t& h, const sparseMatrix_t& G, const vector_t& g,
//                            const sparseMatrix_t& C, const vector_t& c, vector_t& x) {
// for (PDAL_int_t innerIterNum = 0; innerIterNum < settings_.maxInnerIter; ++innerIterNum) {
//   // L = H * xResult + h + G ' * y_ - (Ic*C)' * w_;
//   vector_t L = h;
//   L.noalias() += H * x;
//   L.noalias() += G.transpose() * x;
//   sparseMatrix_t temp = Ic * C;
//   L.noalias() -= temp.transpose() * w_;

//   PDAL_int_t sizeHessian = numDecisionVariables_ + numEqConstraints + numIneqConstraints_;
//   vector_t residual(sizeHessian);
//   residual << L, eqConstraints_ + 1 / rho_ * (lambda_ - y_), Ic * ineqConstraints_ + 1 / rho_ * (mu_ - w_);
//   if (residual.norm() < settings_.innerTolerance) {
//     break;
//   }

//   PDAL_int_t nnzHessian = (sizeHessian + 1) * sizeHessian / 2;
//   std::vector<triplet_t> hessianTriplets;
//   hessianTriplets.reserve(nnzHessian);
//   emplaceBackMatrixToTriplet(0, 0, H.triangularView<Eigen::Upper>(), hessianTriplets);
//   emplaceBackMatrixToTriplet(0, numDecisionVariables_, G.transpose(), hessianTriplets);
//   emplaceBackMatrixToTriplet(0, numDecisionVariables_ + numEqConstraints, (-Ic * C).transpose(), hessianTriplets);
//   emplaceRepeatDiagonalToTriplet(numDecisionVariables_, -1.0 / rho_, numEqConstraints + numIneqConstraints_,
//                                  hessianTriplets);
// }
// }
}  // namespace pdal