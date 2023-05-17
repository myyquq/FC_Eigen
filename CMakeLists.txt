cmake_minimum_required(VERSION 3.25)
project(MNIST)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_TOOLCHAIN_FILE "C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    set(OpenCV_DIR "C:/dev/vcpkg/buildtrees/opencv4/x64-windows-dbg")
elseif(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
    if(MSVC)
        set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
        set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} /O2 /Ob2 /GL /Gy /EHsc /fp:fast /arch:AVX2 /MP /openmp /source-charset:utf-8")
    else()
        set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -O3 -mavx -mfma -ffast-math")
    endif()
    set(OpenCV_DIR "C:/dev/vcpkg/buildtrees/opencv4/x64-windows-rel")
else()
    message("Unknown build type:${CMAKE_BUILD_TYPE}")
endif()

set(EIGEN3_INCLUDE_DIR Eigen)

add_subdirectory(
        fmt
)

#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(OpenCV REQUIRED HINTS ${OpenCV_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Threads REQUIRED)

file(GLOB HEADERS "*.hpp" "*.h")
add_executable(
        MNIST
        main.cpp
        ${HEADERS}
)

target_link_libraries(
        MNIST
        PUBLIC
        fmt::fmt
        ${OpenCV_LIBS}
        Threads::Threads
)
