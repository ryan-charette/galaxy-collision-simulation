# Galaxy-Collision-Simulation

This project is a Python-based simulation of galactic dynamics, focusing on galactic collisions, stellar clusters, and the expansion of the universe. It includes modules for setting up initial conditions, solving differential equations using numerical methods, and visualizing the results in 2D and 3D.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Visualization](#visualization)
- [References](#references)

---

## Features

1. **Galactic Collision Simulation**:
   - Simulates collisions between galaxies using the Leapfrog integration method.
   - Models gravitational interactions between stars and galactic centers.
   - Visualizes the collision process in 2D and 3D.

2. **Stellar Cluster Analysis**:
   - Loads and processes snapshot data of stellar clusters.
   - Analyzes distributions of star masses, velocities, and positions.
   - Generates histograms and evolution plots for the cluster.

3. **Cosmological Expansion**:
   - Implements the Friedmann equations to model the expansion of the universe.
   - Supports multiple cosmological models (standard, sub-critical, critical, super-critical).
   - Plots the evolution of the scale factor over time.

---

## Project Structure

```
├── driver_galcol.ipynb # Jupyter notebook for step-by-step simulation 
├── galaxy_formation.py # Script for generating galaxy formation scenarios 
├── galcol.py # Core module for simulating galactic collisions 
├── galcol_driver.py # Example script to run galcol simulations 
```

---

## Requirements

- Python 3.9+
- Libraries:
  - `numpy`
  - `matplotlib`
  - `astropy`
  - `scipy`

You can install the necessary libraries using:

```bash
pip install numpy matplotlib astropy scipy
```

---

## Installation
Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/ryan-charette/galaxy-collision-simulation.git
cd galaxy-collision-simulation
```

---

## Usage

  To simulate a collision between two disk galaxies:

   ```python
    import galcol
    import astropy.units as unit
    
    # Define galaxy parameters
    galaxies = {
        'intruder': galcol.parameters(1e10, 5, (25, -25, -5), (-75, 75, 0), (0, 0, 1), 5, 1000, 0.025),
        'target': galcol.parameters(5e10, 10, (-5, 5, 1), (15, -15, 0), (1, -1, 2**0.5), 10, 4000, 0.025),
    }
    
    # Initialize galaxy disks
    galcol.init_disk(galaxies['intruder'])
    galcol.init_disk(galaxies['target'])
    
    # Run the simulation
    t, data = galcol.evolve_two_disks(galaxies['target'], galaxies['intruder'], N_steps=10000, N_snapshots=500, time_step=0.05 * unit.Myr)
    
    # Visualize a snapshot in 3D
    galcol.show_two_disks_3d(data[100, :, :], galaxies['target']['N_stars'], [-15, 15], [-15, 15], [-15, 15], t[100], name='two_disks')
```

To analyze stellar clusters (data not included):
```python
import numpy as np

# Load snapshot data
data = np.loadtxt("data_files/nbody/output_0.dat", dtype='float64')

# Compute average mass
average_mass = np.mean(data[:, 1])
print(f"Average mass: {average_mass:.3f} solar masses")
```

To model the expansion of the universe:
```python
import numpy as np
import astropy.units as unit
from numkit import rk4_step

H0 = 67.7 * unit.km / unit.s / unit.Mpc

# Define cosmology
cosmology = {
    'standard': (0.309, 1e-5, 0.691),
    'matter sub': (0.309, 1e-5, 0),
    'matter crit': (1, 0, 0),
    'matter super': (2, 0, 0),
}

# Solve and plot results
# (Refer to `driver_galcol.ipynb` for full implementation)
```

---

## Visualization

- 2D and 3D plots for galaxy collisions.
- Histograms for stellar mass and velocity distributions.
- Scale factor evolution for different cosmological models.
  
Generated plots are saved in the working directory as PDFs or GIFs.

---

## References

- Schroder & Comins, Astronomy 1988/12, 90-96
- Astropy Documentation: https://docs.astropy.org
- NumPy Documentation: https://numpy.org/doc/

--- 

For more details, refer to the `driver_galcol.ipynb` notebook.
