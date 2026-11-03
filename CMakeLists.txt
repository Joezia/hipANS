cmake_minimum_required(VERSION 3.10)

# Allow cluster/user environments to provide hipcc outside /usr/bin.
if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(HIPCC_EXECUTABLE NAMES hipcc HINTS /opt/rocm/bin)
    if(NOT HIPCC_EXECUTABLE)
        message(FATAL_ERROR "hipcc was not found; provide -DCMAKE_CXX_COMPILER=/path/to/hipcc")
    endif()
    set(CMAKE_CXX_COMPILER "${HIPCC_EXECUTABLE}")
endif()
project(hipANS LANGUAGES HIP CXX VERSION 1.0)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm" CACHE PATH "Path to which ROCm/HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
find_package(HIP REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(HIPANS_LIBRARY_TYPE "SHARED" CACHE STRING "Build hipANS component libraries as SHARED or STATIC")
set_property(CACHE HIPANS_LIBRARY_TYPE PROPERTY STRINGS SHARED STATIC)

# Decode optimizations (see docs/OPTIMIZATIONS.md). Defaults keep upstream behaviour.
set(HIPANS_BLOCK_SIZE 8192 CACHE STRING "ANS block size in bytes: 8192, 4096, 2048 or 1024 (O3)")
set(HIPANS_DECODE_WARP_BARRIER 1 CACHE STRING "1 keeps the barrier in decodeOneWarp, 0 removes it (O1)")
set(HIPANS_OPT_DEFINITIONS "-DHIPANS_BLOCK_SIZE=${HIPANS_BLOCK_SIZE} -DHIPANS_DECODE_WARP_BARRIER=${HIPANS_DECODE_WARP_BARRIER}")
string(APPEND CMAKE_CXX_FLAGS " ${HIPANS_OPT_DEFINITIONS}")
string(APPEND CMAKE_HIP_FLAGS " ${HIPANS_OPT_DEFINITIONS}")

add_subdirectory(third_party/glog)
add_subdirectory(third_party/googletest)

add_subdirectory(hipans/utils)
add_subdirectory(hipans/ans)

option(HIPANS_BUILD_BENCH "Build the ANS-to-LDS decode benchmark" ON)
if(HIPANS_BUILD_BENCH)
    add_subdirectory(hipans/bench)
endif()

option(HIPANS_BUILD_FULL_LIBRARY "Build float codec, PyTorch API, and benchmarks" ON)
if(HIPANS_BUILD_FULL_LIBRARY)
    add_subdirectory(hipans/float)
    add_subdirectory(hipans)
endif()
