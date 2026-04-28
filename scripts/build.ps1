$ErrorActionPreference = "Stop"

$BuildDir = if ($env:BUILD_DIR) { $env:BUILD_DIR } else { "build" }
$BuildType = if ($env:BUILD_TYPE) { $env:BUILD_TYPE } else { "Release" }
$EnableMpi = if ($env:ENABLE_MPI) { $env:ENABLE_MPI } else { "ON" }
$EnableCuda = if ($env:ENABLE_CUDA) { $env:ENABLE_CUDA } else { "ON" }

cmake -S . -B $BuildDir -DCMAKE_BUILD_TYPE=$BuildType -DENABLE_MPI=$EnableMpi -DENABLE_CUDA=$EnableCuda
cmake --build $BuildDir --config $BuildType
