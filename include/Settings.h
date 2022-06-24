#pragma once

#include "Types.h"

namespace pdal {
struct Settings {
  PDAL_float_t initialRho = 10.0;
  PDAL_int_t maxOuterIter = 50;
  PDAL_int_t maxInnerIter = 10;
  PDAL_float_t outerTolerance = 1e-6;
  PDAL_float_t innerTolerance = 1e-6;
};

}  // namespace pdal