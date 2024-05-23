cmake_minimum_required(VERSION 3.14)

project(EProjSimplex)
set(BIN_NAME EProjSimplex_native)

set(CMAKE_CXX_STANDARD 17)

set(Eigen3_DIR eigen/share/eigen3/cmake)
find_package(Eigen3 REQUIRED NO_MODULE NO_DEFAULT_PATH)

find_package(Matlab REQUIRED)
find_package(OpenMP REQUIRED)

matlab_add_mex(
  NAME ${BIN_NAME}
  SRC EProjSimplex_native.cpp
  LINK_TO Eigen3::Eigen
  OpenMP::OpenMP_CXX R2018a)

# Matlab has builtin MKL BLAS
target_compile_definitions(${BIN_NAME} PRIVATE EIGEN_USE_BLAS)

add_custom_command(
  TARGET ${BIN_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${BIN_NAME}>
          ${CMAKE_SOURCE_DIR})
