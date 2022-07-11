set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.6.1.tar.gz
)
FetchContent_MakeAvailable(googlebenchmark)

# #############
# ## Targets ##
# #############
add_executable(BenchmarkPDAL
  data/TestDataBenchmark.cpp
  BenchmarkPDAL.cpp
)
target_link_libraries(BenchmarkPDAL
  ${PROJECT_NAME}
  benchmark::benchmark_main
)
target_include_directories(BenchmarkPDAL PRIVATE data)

add_executable(TestPDAL
  data/TestDataBenchmark.cpp
  TestPDAL.cpp
)
target_link_libraries(TestPDAL
  ${PROJECT_NAME}
)
target_include_directories(TestPDAL PRIVATE data)


find_package(osqp QUIET)
find_package(OsqpEigen QUIET)
if(osqp_FOUND AND OsqpEigen_FOUND)
message(STATUS "Found osqp and OsqpEigen")
add_executable(BenchmarkOSQP
  data/TestDataBenchmark.cpp
  BenchmarkOSQP.cpp
)
target_link_libraries(BenchmarkOSQP
  ${PROJECT_NAME}
  ${osqp_LIBRARIES}
  OsqpEigen::OsqpEigen
  benchmark::benchmark_main
)
target_include_directories(BenchmarkOSQP PRIVATE data)


add_executable(TestOSQP
  data/TestDataBenchmark.cpp
  TestOSQP.cpp
)
target_link_libraries(TestOSQP
  ${PROJECT_NAME}
  ${osqp_LIBRARIES}
  OsqpEigen::OsqpEigen
)
target_include_directories(TestOSQP PRIVATE data)

elseif()
message(WARNING "To benchmark against the osqp, please install osqp and osqp-eigen.")
endif()