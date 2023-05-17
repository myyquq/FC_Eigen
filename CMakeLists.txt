cmake_minimum_required(VERSION 3.25)
project(MNIST)

set(CMAKE_CXX_STANDARD 23)

if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS}")
elseif(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -O3 -mavx -ffast-math")
else()
    message("Unknown build type:${CMAKE_BUILD_TYPE}")
endif()

set(EIGEN3_INCLUDE_DIR Eigen)

add_subdirectory(
        fmt
)

#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

file(GLOB HEADERS "*.hpp" "*.h")
add_executable(
        MNIST
        main.cpp
        ${HEADERS}
        painter.hpp)

target_link_libraries(
        MNIST
        fmt::fmt
        ${OpenCV_LIBS}
)
