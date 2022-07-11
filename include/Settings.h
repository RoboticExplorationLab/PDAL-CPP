#pragma once

#include "Types.h"

namespace pdal {
struct Settings {
  PDAL_float_t initialRho = 10.0;
  PDAL_int_t maxOuterIter = 50;
  PDAL_int_t maxInnerIter = 10;

  PDAL_float_t dualResidualTolerance = 1e-8;
  PDAL_float_t primalResidualTolerance = 1e-6;
  PDAL_float_t innerTolerance = 1e-6;

  PDAL_float_t amplificationRho = 2.0;

  PDAL_bool_t displayShortSummary = 1;
  PDAL_bool_t displayRunTime = 1;
};

}  // namespace pdal