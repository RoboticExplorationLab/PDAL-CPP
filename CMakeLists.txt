cmake_minimum_required(VERSION 3.5.0)
project(pdal VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "")

# QDLDL config
set(QDLDL_BUILD_SHARED_LIB OFF CACHE INTERNAL "")
set(QDLDL_BUILD_DEMO_EXE OFF CACHE INTERNAL "")
set(QDLDL_UNITTESTS OFF CACHE INTERNAL "")
set(QDLDL_LONG OFF CACHE INTERNAL "")
set(QDLDL_FLOAT OFF CACHE INTERNAL "")

set(BUILD_BENCHMARKING ON)
set(BUILD_TESTING ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "-Wall -Wfatal-errors")
set(CMAKE_CXX_FLAGS "-Wall -Wfatal-errors")

include(FetchContent)

find_package(Eigen3 REQUIRED NO_MODULE)
add_subdirectory(qdldl)

include_directories(
  include
)

# ###########
# ## Build ##
# ###########
add_library(${PROJECT_NAME}
  src/Pdal.cpp
)
target_link_libraries(${PROJECT_NAME}
  qdldlstatic
  Eigen3::Eigen
)

# ##################
# ## Benchmarking ##
# ##################
if(BUILD_BENCHMARKING)
  add_subdirectory(benchmark)
endif()

# ##########
# ## Test ##
# ##########
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()