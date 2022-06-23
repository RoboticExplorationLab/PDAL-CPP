cmake_minimum_required(VERSION 3.0.0)
project(pdal VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED NO_MODULE)
add_subdirectory(qdldl)

include_directories(
  include
)


add_library(${PROJECT_NAME} src/Pdal.cpp)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
