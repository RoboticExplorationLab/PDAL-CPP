#pragma once

#include "Types.h"

namespace pdal {
struct Settings {
  PDAL_float_t initialRho = 10.0;

  PDAL_int_t maxOuterIter = 50;
  PDAL_int_t maxInnerIter = 10;

  PDAL_float_t primalResidualAbsoluteTolerance = 1e-3;
  PDAL_float_t primalResidualRelativeTolerance = 1e-3;

  PDAL_float_t dualResidualAbsoluteTolerance = 1e-3;
  PDAL_float_t dualResidualRelativeTolerance = 1e-3;

  PDAL_float_t newtonAbsoluteTolerance = 1e-3;
  PDAL_float_t newtonRelativeTolerance = 1e-3;

  PDAL_float_t amplificationRho = 2.0;

  PDAL_bool_t displayShortSummary = 1;
  PDAL_bool_t displayRunTime = 1;
};

}  // namespace pdal