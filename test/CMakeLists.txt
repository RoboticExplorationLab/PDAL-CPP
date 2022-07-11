FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# #############
# ## Targets ##
# #############
add_executable(pdal-test
  TestPdal.cpp
  TestData.cpp
  TestData.h
)
target_link_libraries(pdal-test
  ${PROJECT_NAME}
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(pdal-test)
