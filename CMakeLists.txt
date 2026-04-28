cmake_minimum_required(VERSION 3.20)

project(DistributedFMMGalaxySim
    VERSION 0.1.0
    DESCRIPTION "Distributed FMM N-body simulator for galaxy collision experiments"
    LANGUAGES CXX
)

option(ENABLE_MPI "Enable MPI support when available" ON)
option(ENABLE_CUDA "Enable CUDA support when available" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)

set(HAVE_MPI 0)
if(ENABLE_MPI)
    find_package(MPI QUIET)
    if(MPI_CXX_FOUND)
        set(HAVE_MPI 1)
        message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
    else()
        message(WARNING "ENABLE_MPI=ON but MPI was not found. Building serial fallback.")
    endif()
endif()

set(HAVE_CUDA 0)
if(ENABLE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(HAVE_CUDA 1)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
    else()
        message(WARNING "ENABLE_CUDA=ON but CUDA was not found. Building CPU-only fallback.")
    endif()
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/build_config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/build_config.hpp
)

add_library(fmm_galaxy_core
    cpp/core/config.cpp
    cpp/core/diagnostics.cpp
    cpp/core/initial_conditions.cpp
    cpp/core/integrator.cpp
    cpp/core/simulation_info.cpp
    cpp/direct/direct_solver.cpp
    cpp/fmm/quadtree.cpp
    cpp/io/snapshot_writer.cpp
)

target_include_directories(fmm_galaxy_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp
        ${CMAKE_CURRENT_BINARY_DIR}/generated
)

if(HAVE_MPI)
    target_link_libraries(fmm_galaxy_core PUBLIC MPI::MPI_CXX)
    target_compile_definitions(fmm_galaxy_core PUBLIC FMM_GALAXY_HAVE_MPI=1)
endif()

if(HAVE_CUDA)
    add_library(fmm_galaxy_cuda STATIC
        cpp/cuda/cuda_smoke.cu
    )
    target_include_directories(fmm_galaxy_cuda
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp
            ${CMAKE_CURRENT_BINARY_DIR}/generated
    )
    target_compile_definitions(fmm_galaxy_cuda PUBLIC FMM_GALAXY_HAVE_CUDA=1)
endif()

add_executable(fmm_galaxy_sim
    cpp/core/main.cpp
)

target_link_libraries(fmm_galaxy_sim PRIVATE fmm_galaxy_core)

add_executable(fmm_galaxy_smoke
    cpp/core/smoke_main.cpp
)

target_link_libraries(fmm_galaxy_smoke PRIVATE fmm_galaxy_core)

if(HAVE_CUDA)
    target_link_libraries(fmm_galaxy_smoke PRIVATE fmm_galaxy_cuda)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(fmm_galaxy_tests
        cpp/tests/smoke_tests.cpp
    )
    target_link_libraries(fmm_galaxy_tests PRIVATE fmm_galaxy_core)
    add_test(NAME smoke_tests COMMAND fmm_galaxy_tests)
endif()

message(STATUS "Build summary:")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  MPI enabled:  ${HAVE_MPI}")
message(STATUS "  CUDA enabled: ${HAVE_CUDA}")
