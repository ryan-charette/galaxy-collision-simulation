# Contributing

Thanks for helping improve the galaxy collision simulator. Contributions should
keep the project focused on reproducible scientific computing: clear numerical
assumptions, validated solver behavior, traceable outputs, and maintainable
C++/Python tooling.

## Development Setup

Install the Python tooling in editable mode:

```bash
python -m pip install -e ".[dev,test,docs]"
```

Build the simulator:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_MPI=ON -DENABLE_CUDA=ON
cmake --build build --parallel
```

If MPI or CUDA are unavailable, CMake builds the available CPU fallback paths.

## Checks

Run the focused checks before opening a pull request:

```bash
python -m compileall -q scripts src/python
python -m pytest
ctest --test-dir build --output-on-failure
```

For documentation changes, also run:

```bash
nox -s docs
```

The docs build requires Doxygen for the generated C++ API reference. If Doxygen
is not on `PATH`, set `DOXYGEN_EXECUTABLE` to the `doxygen` binary.

## Numerical Changes

Solver, integrator, CUDA, MPI, and diagnostics changes should include one or
more of:

- direct-solver comparison for small particle counts
- conservation diagnostics for energy, momentum, or angular momentum
- convergence or sensitivity notes for timestep, softening, or opening angle
- benchmark updates when performance behavior changes

When changing benchmark results, record the hardware, compiler, build type,
solver settings, repetitions, and whether timings include I/O.

## Pull Requests

Keep pull requests scoped. Include:

- what changed
- why it changed
- how it was validated
- any expected numerical or performance impact

Do not commit generated build directories, local virtual environments, cache
files, or large experiment artifacts unless they are intentionally curated
documentation assets.
