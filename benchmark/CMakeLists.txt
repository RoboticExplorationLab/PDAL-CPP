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