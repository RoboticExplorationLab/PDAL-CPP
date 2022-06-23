cmake_minimum_required(VERSION 3.0.0)
project(pdal VERSION 0.1.0)

find_package(Eigen3 REQUIRED)
add_library(pdal pdal.cpp)
