cmake_minimum_required(VERSION 3.0.0)
project(pdal VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# QDLDL config
set(QDLDL_BUILD_SHARED_LIB OFF)
set(QDLDL_BUILD_DEMO_EXE OFF)
set(QDLDL_UNITTESTS OFF) 
set(QDLDL_LONG OFF)

set(CMAKE_C_FLAGS "-Wall -Wfatal-errors")
set(CMAKE_CXX_FLAGS "-Wall -Wfatal-errors")

find_package(Eigen3 REQUIRED NO_MODULE)
add_subdirectory(qdldl)

include_directories(
  include
)
add_library(${PROJECT_NAME}
  src/Pdal.cpp
)
target_link_libraries(${PROJECT_NAME} 
  qdldlstatic
  Eigen3::Eigen
)

###########
### Test ##
###########

enable_testing()
include(GoogleTest)
find_package(GTest REQUIRED)

add_executable(${PROJECT_NAME}-test
  test/TestPdal.cpp
  test/TestData.cpp
  test/TestData.h
)
target_link_libraries(${PROJECT_NAME}-test
  ${PROJECT_NAME}
  qdldlstatic
  Eigen3::Eigen
  GTest::gtest_main
)
gtest_discover_tests(${PROJECT_NAME}-test)