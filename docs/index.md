# Galaxy Collision Simulation Documentation

This documentation covers the C++/CUDA galaxy collision simulator, Python
analysis tools, benchmark workflow, and machine-learning support utilities.

The simulator models softened Newtonian gravity for galaxy collision
experiments. It includes direct summation, Barnes-Hut treecode, FMM, MPI, and
optional CUDA execution paths. Python tools provide reproducible experiment
runners, snapshot loading, plotting, animation, benchmark analysis, ML dataset
generation, baseline supervised models, and adaptive solver tuning utilities.

```{toctree}
:maxdepth: 2
:caption: User Guide

design
architecture
testing
benchmarks/index
```

```{toctree}
:maxdepth: 2
:caption: Machine Learning

ml_datasets
ml_models
rl_environment
error_correction
```

```{toctree}
:maxdepth: 2
:caption: Project History

milestones
```

## Quick Commands

Build and test the C++ simulator:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_MPI=ON -DENABLE_CUDA=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Run a smoke simulation:

```bash
./build/fmm_galaxy_sim --config configs/smoke_test.toml
```

Generate a small force-error benchmark:

```bash
python scripts/run_force_error_benchmarks.py --smoke
```

Build these docs:

```bash
nox -s docs
```

The generated HTML is written to `docs/_build/html`.
