cmake_minimum_required(VERSION 3.12)
project(CVC_Pybind)

# Find PCL and any other dependencies here
find_package(Eigen3 REQUIRED)
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


# Define your C++ library
add_library(clusterlib SHARED CVC_cluster.cpp)  # Include your existing C++ code
target_link_libraries(clusterlib Eigen3::Eigen)

# Link against any necessary libraries, such as PCL

# Add the Python bindings file
pybind11_add_module(pycluster pybinding.cpp)

# Link your Python module against your C++ library
target_link_libraries(pycluster PRIVATE clusterlib)
