"""
Plot the result of Kelvin-Helmholtz instability simulation.

Usage:
    python animate_kelvin_helmholtz_2d.py

Author: Ching-Yin Ng
Date: 2025-03-22
"""

import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import PIL
from matplotlib.colors import LinearSegmentedColormap

SNAPSHOT_FOLDER = Path(__file__).parent / "snapshots/"
FRAME_FOLDER = Path(__file__).parent / "frames/"
FILE_PATH = Path(__file__).parent / "kelvin_helmholtz_2d.gif"

LENGTH = 35.0

colors = ["#145c8c", "#1f78b4", "#ffffff", "#e02225", "#821012"]
cmap_name = "custom"

# Create the custom colormap
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
CMAP = custom_cmap
# CMAP='coolwarm'


def main() -> None:
    FRAME_FOLDER.mkdir(exist_ok=True)

    # Load the snapshot
    def natural_sort_key(s):
        """Sort strings with embedded numbers naturally."""
        return [int(Path(s).stem.strip("snapshot_"))]

    snapshot_files = sorted(
        glob.glob(str(SNAPSHOT_FOLDER / "snapshot_*.h5")), key=natural_sort_key
    )

    for i, snapshot_file in enumerate(snapshot_files):
        with h5py.File(snapshot_file, "r") as f:
            density = f["fields/density"][()]

            x_min = f["parameters/x_min"][()]
            x_max = f["parameters/x_max"][()]
            y_min = f["parameters/y_min"][()]
            y_max = f["parameters/y_max"][()]
            num_cells_x = f["parameters/num_cells_x"][()]
            num_cells_y = f["parameters/num_cells_y"][()]
            num_ghost_cells_side = f["parameters/num_ghost_cells_side"][()]

            t = f["simulation_status/simulation_time"][()]

            # Drawing frame
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(
                density.reshape(
                    num_cells_x + 2 * num_ghost_cells_side,
                    num_cells_y + 2 * num_ghost_cells_side,
                )[
                    num_ghost_cells_side:-num_ghost_cells_side,
                    num_ghost_cells_side:-num_ghost_cells_side,
                ],
                origin="lower",
                extent=(x_min, x_max, y_min, y_max),
                vmin=1.0,
                vmax=2.0,
                cmap=CMAP,
            )
            colorbar = plt.colorbar(ax.images[0], ax=ax, shrink=0.75)
            colorbar.set_label("Density")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Kelvin-Helmholtz Instability (t={t:.2f})")

            plt.tight_layout()
            
            plt.savefig(FRAME_FOLDER / f"frame_{i:04d}.png")
            plt.close("all")

    def frames_generator(n_frames: int):
        for i in range(n_frames):
            yield PIL.Image.open(FRAME_FOLDER / f"frame_{i:04d}.png")

    frames = frames_generator(len(snapshot_files))
    next(frames).save(
        FILE_PATH,
        save_all=True,
        append_images=frames,
        loop=0,
        duration=(1000 // (len(snapshot_files) / LENGTH)),
    )

    for i in range(len(snapshot_files)):
        (FRAME_FOLDER / f"frame_{i:04d}.png").unlink()

    FRAME_FOLDER.rmdir()

    print(f"Output completed! Please check {FILE_PATH}")
    print()


if __name__ == "__main__":
    main()
