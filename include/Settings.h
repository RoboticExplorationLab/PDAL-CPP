#pragma once

#include "Types.h"

namespace pdal {
struct Settings {
  PDAL_float_t initialRho = 10.0;
  PDAL_int_t maxOuterIter = 50;
  PDAL_int_t maxInnerIter = 10;
};

}  // namespace pdal