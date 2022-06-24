#pragma once

#include <qdldl_types.h>

#include "Types.h"

extern const QDLDL_int An;
extern const QDLDL_int Ap[];
extern const QDLDL_int Ai[];
extern const QDLDL_float Ax[];

pdal::sparseMatrix_t get_H();
pdal::sparseMatrix_t get_G();
pdal::sparseMatrix_t get_C();
pdal::vector_t get_h();
pdal::vector_t get_g();
pdal::vector_t get_c();