$ErrorActionPreference = "Stop"

$BuildDir = if ($env:BUILD_DIR) { $env:BUILD_DIR } else { "build" }
$BuildType = if ($env:BUILD_TYPE) { $env:BUILD_TYPE } else { "Release" }

$SmokeCandidates = @(
    (Join-Path $BuildDir "fmm_galaxy_smoke.exe"),
    (Join-Path (Join-Path $BuildDir $BuildType) "fmm_galaxy_smoke.exe")
)
$SimCandidates = @(
    (Join-Path $BuildDir "fmm_galaxy_sim.exe"),
    (Join-Path (Join-Path $BuildDir $BuildType) "fmm_galaxy_sim.exe")
)

$SmokeExec = $SmokeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
$SimExec = $SimCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $SmokeExec -or -not $SimExec) {
    throw "Executables not found. Run .\scripts\build.ps1 first."
}

ctest --test-dir $BuildDir --output-on-failure -C $BuildType
& $SmokeExec
& $SimExec --config configs/smoke_test.toml
