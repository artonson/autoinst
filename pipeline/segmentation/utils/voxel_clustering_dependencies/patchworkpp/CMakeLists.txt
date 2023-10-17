cmake_minimum_required(VERSION 3.12)
project(patchworkpp_python_wrapper)


set(CMAKE_CXX_STANDARD 14)
set(PYTHON_EXECUTABLE python3)
set(CMAKE_BUILD_TYPE Release)


find_package(Eigen3 REQUIRED QUIET)
# Add Pybind11
include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.2.3
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

#add_subdirectory(pybind11)


# Define your C++ library
add_library(patchworkpp SHARED patchworkpp.cpp)  # Include your existing C++ code
target_link_libraries(patchworkpp Eigen3::Eigen)
add_library(PATCHWORK::patchworkpp ALIAS patchworkpp)

# Link against any necessary libraries, such as PCL

# Add the Python bindings file
pybind11_add_module(pypatchworkpp pybinding.cpp)

# Link your Python module against your C++ library
target_link_libraries(pypatchworkpp PUBLIC PATCHWORK::patchworkpp)


if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(pypatchworkpp PUBLIC -fsized-deallocation -Ofast)
endif()













