cmake_minimum_required(VERSION 3.29)
project(mpihelper LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI COMPONENTS CXX REQUIRED)


# find_package(GTest CONFIG REQUIRED)
include(FetchContent)
# Specify GoogleTest details (use a specific tag/commit for stability)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG      v1.17.0
)

# Fetch and populate the content
FetchContent_MakeAvailable(googletest)


add_library(MPIHelper INTERFACE)
target_link_libraries(MPIHelper INTERFACE MPI::MPI_CXX)
target_include_directories(MPIHelper INTERFACE src)

add_executable(test tests/test.cc)
target_link_libraries(test PRIVATE MPIHelper GTest::gtest_main)
