import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)  # Ensure LaTeX is installed and configured

VMIN=1.0
VMAX=2.0

# Parameters
PROCESSED_FILES_FOLDER = "processed_files/"
FILES = [
    "snapshots_godunov_256_1.5.csv",
    "snapshots_godunov_256_3.0.csv",
    "snapshots_godunov_256_9.0.csv",
    "snapshots_godunov_256_15.0.csv",
    "snapshots_mhm_256_1.5.csv",
    "snapshots_mhm_256_3.0.csv",
    "snapshots_mhm_256_9.0.csv",
    "snapshots_mhm_256_15.0.csv",
    "snapshots_mhm_512_1.5.csv",
    "snapshots_mhm_512_3.0.csv",
    "snapshots_mhm_512_9.0.csv",
    "snapshots_mhm_512_15.0.csv",
]
colors = ["#145c8c", "#1f78b4", "#ffffff", "#e02225", "#821012"]
cmap_name = "custom"

# Create the custom colormap
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
CMAP = custom_cmap
# CMAP='coolwarm'

# Use a 3x4 grid for 12 panels
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(9, 7))
axs = axs.flatten()

axs[0].set_ylabel("Godunov $256^2$")
axs[4].set_ylabel("MHM $256^2$")
axs[8].set_ylabel("MHM $512^2$")

axs[0].set_title("$t = 1.5$")
axs[1].set_title("$t = 3.0$")
axs[2].set_title("$t = 9.0$")
axs[3].set_title("$t = 15.0$")

im_list = []  # to store image objects for color normalization

for ax, filename in zip(axs, FILES):
    filepath = os.path.join(PROCESSED_FILES_FOLDER, filename)
    # Read the CSV file with polars. Assume columns: x, y, density.
    data = pl.read_csv(filepath)
    # Convert to numpy arrays
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    density = data["density"].to_numpy()
    
    # Determine grid dimensions based on filename resolution.
    if "256" in filename:
        n_x = n_y = 252
    elif "512" in filename:
        n_x = n_y = 508
    else:
        n_x = n_y = 252  # fallback default

    # Reshape density field assuming the CSV rows are ordered appropriately.
    density_grid = density.reshape((n_y, n_x))
    
    # Set proper extent for imshow: [xmin, xmax, ymin, ymax]
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    extent = [x_unique[0], x_unique[-1], y_unique[0], y_unique[-1]]

    # Plot without interpolation
    im = ax.imshow(density_grid, origin='lower', extent=extent, aspect='auto', cmap=CMAP, vmin=VMIN, vmax=VMAX)
    im_list.append(im)

    ax.set_xticks([])
    ax.set_yticks([])

# Adjust subplot spacing so that there is no whitespace
plt.subplots_adjust(wspace=0, hspace=0)

# Create a common colorbar. Set norm using the min/max across all images.
all_data = np.concatenate([im.get_array().ravel() for im in im_list])
vmin, vmax = all_data.min(), all_data.max()

# Update all images to use the same color normalization
# for im in im_list:
#     im.set_clim(vmin, vmax)

fig.colorbar(im_list[0], ax=axs, orientation='vertical', fraction=0.04, pad=0.0)
plt.show()