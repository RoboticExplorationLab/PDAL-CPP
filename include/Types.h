#pragma once

#include <qdldl_types.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace pdal {

using PDAL_float_t = QDLDL_float;
using PDAL_int_t = QDLDL_int;
using PDAL_bool_t = QDLDL_bool;

using sparseMatrix_t = Eigen::SparseMatrix<PDAL_float_t>;
using vector_t = Eigen::Matrix<PDAL_float_t, Eigen::Dynamic, 1>;
using matrix_t = Eigen::Matrix<PDAL_float_t, Eigen::Dynamic, Eigen::Dynamic>;
using triplet_t = Eigen::Triplet<PDAL_float_t>;

}  // namespace pdal