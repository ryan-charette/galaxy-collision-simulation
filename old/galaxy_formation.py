import numpy as np
from astropy.cosmology import Planck15
from astropy.cosmology import z_at_value
from astropy.cosmology import LambdaCDM
import matplotlib.pyplot as plt
from matplotlib import animation

# Set cosmological parameters
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

# Set redshift range
zmin, zmax = 0, 10

# Calculate comoving distance in Mpc
comoving_distance = cosmo.comoving_distance(zmin).value - cosmo.comoving_distance(zmax).value

# Calculate number density of galaxies in Mpc^-3
n_gal = 3e-4 * comoving_distance**3

# Generate random positions for galaxies in Mpc
x_pos = np.random.uniform(-comoving_distance/2, comoving_distance/2, size=n_gal)
y_pos = np.random.uniform(-comoving_distance/2, comoving_distance/2, size=n_gal)
z_pos = np.random.uniform(-comoving_distance/2, comoving_distance/2, size=n_gal)

# Generate random velocities for galaxies in km/s
vx = np.random.normal(0, 100, size=n_gal)
vy = np.random.normal(0, 100, size=n_gal)
vz = np.random.normal(0, 100, size=n_gal)

# Generate random masses for galaxies in solar masses
masses = np.random.uniform(1e8, 1e12, size=n_gal)

# Calculate redshift for each galaxy using its comoving distance
z = z_at_value(cosmo.comoving_distance, np.sqrt(x_pos**2 + y_pos**2 + z_pos**2))

# Set up plot
fig = plt.figure(figsize=(8, 6))
ax = plt.axes(xlim=(-comoving_distance/2, comoving_distance/2), ylim=(-comoving_distance/2, comoving_distance/2))
scatter = ax.scatter([], [])

# Set up animation function
def animate(i):
    # Update galaxy positions based on their velocities
    x_pos += vx * cosmo.lookback_time(zmax - i/10)
    y_pos += vy * cosmo.lookback_time(zmax - i/10)
    z_pos += vz * cosmo.lookback_time(zmax - i/10)

    # Update redshifts based on new positions
    z = z_at_value(cosmo.comoving_distance, np.sqrt(x_pos**2 + y_pos**2 + z_pos**2))

    # Update plot
    scatter.set_offsets(np.stack((x_pos, y_pos), axis=1))

# Run animation
anim = animation.FuncAnimation(fig, animate, frames=100)
plt.show()
