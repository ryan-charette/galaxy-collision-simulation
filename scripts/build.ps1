$ErrorActionPreference = "Stop"

$BuildDir = if ($env:BUILD_DIR) { $env:BUILD_DIR } else { "build" }
$BuildType = if ($env:BUILD_TYPE) { $env:BUILD_TYPE } else { "Release" }

cmake -S . -B $BuildDir -DCMAKE_BUILD_TYPE=$BuildType -DENABLE_MPI=OFF -DENABLE_CUDA=OFF
cmake --build $BuildDir --config $BuildType
