#include "Pdal.h"

#include <gtest/gtest.h>
#include <Eigen/SparseCore>

using namespace pdal;
using sparseMatrix_t = PdalSolver::sparseMatrix_t;
using vector_t = PdalSolver::vector_t;

const QDLDL_int An = 10;
const QDLDL_int Ap[] = {0, 1, 2, 4, 5, 6, 8, 10, 12, 14, 17};
const QDLDL_int Ai[] = {0, 1, 1, 2, 3, 4, 1, 5, 0, 6, 3, 7, 6, 8, 1, 2, 9};
const QDLDL_float Ax[] = {1.0,       0.460641,   -0.121189, 0.417928, 0.177828,  0.1,      -0.0290058, -1.0, 0.350321,
                          -0.441092, -0.0845395, -0.316228, 0.178663, -0.299077, 0.182452, -1.56506,   -0.1};
const QDLDL_float b[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

TEST(Pdal, elimination_tree_construction) {
  Eigen::Map<const sparseMatrix_t> H(An, An, Ap[An], Ap, Ai, Ax);
  PdalSolver solver;
  solver.setupProblem(H, vector_t(), sparseMatrix_t(), vector_t(), sparseMatrix_t(), vector_t());
  std::vector<QDLDL_int> work(An), Lnz(An), etree(An);
  QDLDL_etree(An, Ap, Ai, work.data(), Lnz.data(), etree.data());
  EXPECT_EQ(solver.etree(), etree);
}