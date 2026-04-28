#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
SMOKE_EXEC="${BUILD_DIR}/fmm_galaxy_smoke"
SIM_EXEC="${BUILD_DIR}/fmm_galaxy_sim"

if [[ ! -x "${SMOKE_EXEC}" || ! -x "${SIM_EXEC}" ]]; then
    echo "Executables not found. Run ./scripts/build.sh first." >&2
    exit 1
fi

ctest --test-dir "${BUILD_DIR}" --output-on-failure
"${SMOKE_EXEC}"
"${SIM_EXEC}" --config configs/smoke_test.toml
