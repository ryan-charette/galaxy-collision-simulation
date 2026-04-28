#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
ENABLE_MPI="${ENABLE_MPI:-ON}"
ENABLE_CUDA="${ENABLE_CUDA:-ON}"

cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DENABLE_MPI="${ENABLE_MPI}" -DENABLE_CUDA="${ENABLE_CUDA}"
cmake --build "${BUILD_DIR}" -j
