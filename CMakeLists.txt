cmake_minimum_required(VERSION 2.8)
project(icpSparse)

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

# Find the Python interpreter and include directory
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11 package (you can also use add_subdirectory if you have a copy of pybind11 in your project)
find_package(pybind11 REQUIRED)

#Adding current directory for include
include_directories(${CMAKE_CURRENT_LIST_DIR})

#Adding Eigen
include_directories(${CMAKE_CURRENT_LIST_DIR}/ext/Eigen)

#Adding Nanoflann
include_directories(${CMAKE_CURRENT_LIST_DIR}/ext/NanoFlann)

#Adding projects headers
include_directories(${CMAKE_CURRENT_LIST_DIR}/lib/IcpOptimizer)

#Adding projects libraries
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/lib/IcpOptimizer)

add_library(sparseIcp MODULE pybind.cpp ${CMAKE_CURRENT_LIST_DIR}/lib/IcpOptimizer/IcpOptimizer.cpp)

# Include directories for pybind11, Python, Eigen, and Nanoflann
target_include_directories(sparseIcp PRIVATE ${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${CMAKE_CURRENT_LIST_DIR}/lib/IcpOptimizer)

# Link against the Python library (Python3::Python), pybind11 library, Eigen, and Nanoflann
target_link_libraries(sparseIcp PRIVATE Python3::Python pybind11::module)

# Set the output name and location of the extension module
set_target_properties(sparseIcp PROPERTIES PREFIX "" OUTPUT_NAME "sparseIcp")

# Install the extension module to the appropriate Python site-packages directory
install(TARGETS sparseIcp LIBRARY DESTINATION "${Python3_SITELIB}")