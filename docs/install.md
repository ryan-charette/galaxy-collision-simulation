# Installation

The project has two installable surfaces:

- the C++ simulator, libraries, headers, configs, and CMake package metadata
- the Python analysis, rendering, benchmark, and sweep command-line tools

Use both when you want a complete local development environment. Use only the
C++ install when you only need the simulator executable or C++ targets.

## Python Tools

For development from a checkout:

```bash
python -m pip install -e ".[dev,test,docs]"
```

For regular use from a checkout:

```bash
python -m pip install .
```

This installs console commands:

```text
fmm-galaxy-benchmark
fmm-galaxy-crossover
fmm-galaxy-force-error
fmm-galaxy-parquet
fmm-galaxy-plot
fmm-galaxy-render
fmm-galaxy-render-gif
fmm-galaxy-render-readme-gif
fmm-galaxy-sweep
fmm-galaxy-viewer
```

The older module and script forms still work from a source checkout, for
example `python -m python.analysis.plot_snapshots` and
`python scripts/run_benchmarks.py`.

## C++ Simulator

Configure and build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_MPI=ON -DENABLE_CUDA=ON -DFMM_GALAXY_FETCH_TEST_DEPS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The C++ tests use Catch2. CMake first looks for an installed Catch2 3 package.
The `FMM_GALAXY_FETCH_TEST_DEPS` option lets CMake download it for local test
builds when it is not installed. For install-only builds that do not run tests,
configure with `-DBUILD_TESTING=OFF`.

Install to a local prefix:

```bash
cmake --install build --prefix install
```

The install tree contains:

```text
install/bin/fmm_galaxy_sim
install/bin/fmm_galaxy_smoke
install/include/fmm_galaxy/
install/lib/cmake/FmmGalaxy/
install/share/fmm-galaxy/configs/
install/share/doc/DistributedFMMGalaxySim/
```

On Windows multi-config generators, pass the build configuration:

```powershell
cmake --install build --config Release --prefix install
```

## CMake Package Consumers

The install exports CMake targets under the `FmmGalaxy::` namespace. A downstream
CMake project can consume the core library with:

```cmake
find_package(FmmGalaxy CONFIG REQUIRED)

add_executable(my_analysis main.cpp)
target_link_libraries(my_analysis PRIVATE FmmGalaxy::fmm_galaxy_core)
```

Point CMake at a local install with:

```bash
cmake -S consumer -B consumer-build -DCMAKE_PREFIX_PATH=/path/to/install
```

Public headers are installed under `include/fmm_galaxy`, and the exported target
adds that include directory automatically.

## Runtime Notes

Parquet snapshot output requires the Python tools and `pyarrow`. If the
simulator should use a specific Python interpreter for conversion, set:

```bash
export FMM_GALAXY_PYTHON=/path/to/python
```

MPI and CUDA are optional at configure time. If requested but unavailable, CMake
builds the available CPU fallback paths and records the requested/available
state in run metadata.
