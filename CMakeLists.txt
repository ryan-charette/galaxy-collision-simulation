cmake_minimum_required(VERSION 3.20)

project(DistributedFMMGalaxySim
    VERSION 0.1.0
    DESCRIPTION "Distributed FMM N-body simulator for galaxy collision experiments"
    LANGUAGES CXX
)

option(ENABLE_MPI "Enable MPI support when available" ON)
option(ENABLE_CUDA "Enable CUDA support when available" ON)

if(ENABLE_MPI)
    set(FMM_GALAXY_CMAKE_ENABLE_MPI 1)
else()
    set(FMM_GALAXY_CMAKE_ENABLE_MPI 0)
endif()

if(ENABLE_CUDA)
    set(FMM_GALAXY_CMAKE_ENABLE_CUDA 1)
else()
    set(FMM_GALAXY_CMAKE_ENABLE_CUDA 0)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)
find_package(OpenMP QUIET)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/core/build_config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/build_config.hpp
)

add_library(fmm_galaxy_core
    src/cpp/core/config.cpp
    src/cpp/core/diagnostics.cpp
    src/cpp/core/initial_conditions.cpp
    src/cpp/core/integrator.cpp
    src/cpp/core/provenance.cpp
    src/cpp/core/simulation_runner.cpp
    src/cpp/core/simulation_info.cpp
    src/cpp/cuda/cuda_solver_stub.cpp
    src/cpp/direct/direct_solver.cpp
    src/cpp/fmm/fmm_solver.cpp
    src/cpp/fmm/multipole.cpp
    src/cpp/fmm/quadtree.cpp
    src/cpp/fmm/tree_geometry.cpp
    src/cpp/io/parquet_converter.cpp
    src/cpp/io/snapshot_writer.cpp
    src/cpp/mpi/distributed_solver.cpp
)

target_include_directories(fmm_galaxy_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
        ${CMAKE_CURRENT_BINARY_DIR}/generated
)

if(HAVE_MPI)
    target_link_libraries(fmm_galaxy_core PUBLIC MPI::MPI_CXX)
    target_compile_definitions(fmm_galaxy_core PUBLIC FMM_GALAXY_HAVE_MPI=1)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(fmm_galaxy_core PUBLIC OpenMP::OpenMP_CXX)
elseif(MSVC)
    target_compile_options(fmm_galaxy_core PUBLIC /openmp)
endif()

if(HAVE_CUDA)
    add_library(fmm_galaxy_cuda STATIC
        src/cpp/cuda/cuda_solver.cu
        src/cpp/cuda/cuda_smoke.cu
    )
    target_include_directories(fmm_galaxy_cuda
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
            ${CMAKE_CURRENT_BINARY_DIR}/generated
    )
    target_link_libraries(fmm_galaxy_cuda PUBLIC fmm_galaxy_core)
    target_compile_definitions(fmm_galaxy_cuda PUBLIC FMM_GALAXY_HAVE_CUDA=1)
endif()

add_executable(fmm_galaxy_sim
    src/cpp/core/main.cpp
)

target_link_libraries(fmm_galaxy_sim PRIVATE fmm_galaxy_core)

add_executable(fmm_galaxy_smoke
    src/cpp/core/smoke_main.cpp
)

target_link_libraries(fmm_galaxy_smoke PRIVATE fmm_galaxy_core)

if(HAVE_CUDA)
    target_link_libraries(fmm_galaxy_sim PRIVATE fmm_galaxy_cuda)
    target_link_libraries(fmm_galaxy_smoke PRIVATE fmm_galaxy_cuda)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(fmm_galaxy_tests
        src/cpp/tests/config_snapshot_tests.cpp
        src/cpp/tests/cuda_fallback_tests.cpp
        src/cpp/tests/math_direct_tests.cpp
        src/cpp/tests/smoke_tests.cpp
        src/cpp/tests/tree_fmm_accuracy_tests.cpp
    )
    target_link_libraries(fmm_galaxy_tests PRIVATE fmm_galaxy_core)
    if(HAVE_CUDA)
        target_link_libraries(fmm_galaxy_tests PRIVATE fmm_galaxy_cuda)
    endif()
    add_test(NAME smoke_tests COMMAND fmm_galaxy_tests)
endif()

message(STATUS "Build summary:")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  MPI enabled:  ${HAVE_MPI}")
message(STATUS "  CUDA enabled: ${HAVE_CUDA}")
message(STATUS "  OpenMP:       ${OpenMP_CXX_FOUND}")
