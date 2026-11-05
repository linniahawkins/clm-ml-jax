"""
Run CLM-ML-JAX on the 1° CLM grid (default: the whole globe, the first 10 days of 2000).

The first time, the script writes an experiment folder from the preset `grid_1deg_2000`
(CLM 5.4 surface data at 0.9x1.25°, CRUJRA meteorology, CTSM 5.4 monthly soil water, CLM6
parameters; inputs on NCAR's Glade storage), then runs it and maps the daily mean gridcell
fluxes. Each gridcell's patches (one per natural PFT) are run independently and combined into
area-weighted gridcell means.

    python examples/run_global.py                                   # whole globe, 10 days
    python examples/run_global.py --region 35 40 -100 -95 --length "1 day"   # a small box
    python examples/run_global.py --devices 4                       # four GPUs

The whole globe (about 72 600 patches on 15 900 gridcells) needs a GPU: a 10-day chunk takes
about 15 minutes on one A100, and more than a day on a CPU. On Derecho, from a GPU node:

    qsub -I -A <account> -q develop -l select=1:ncpus=16:ngpus=1:mem=200GB -l walltime=01:00:00
    python examples/run_global.py

A small box runs on a CPU: `--region 35 40 -100 -95 --length "1 day"` (25 gridcells, 130
patches) takes under a minute. The same run from the command line:
`clm-ml run grid_example --set domain.grid.region=global`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import clm_ml

MAPPED = ("latent_heat_flux", "sensible_heat_flux", "gpp", "canopy_top_air_temperature")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--folder", default="grid_example", help="experiment folder")
    parser.add_argument(
        "--region", nargs=4, type=float, metavar=("SOUTH", "NORTH", "WEST", "EAST"),
        help="a latitude-longitude box in degrees (default: the whole globe)",
    )  # fmt: skip
    parser.add_argument("--length", default="10 days", help='run length (default "10 days")')
    parser.add_argument("--devices", type=int, default=1, help="GPUs to spread patches over")
    parser.add_argument(
        "--set", action="append", default=[], metavar="FILE.KEY=VALUE",
        help="change a setting without editing the files (repeatable), as `clm-ml run --set`",
    )  # fmt: skip
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        clm_ml.new_experiment(folder, "grid_1deg_2000")
        print(f"wrote {folder}/ from the preset grid_1deg_2000")

    if args.region:
        s, n, w, e = args.region
        region = f"{{lat: [{s}, {n}], lon: [{w}, {e}]}}"
    else:
        region = "global"
    overrides = [
        f"domain.grid.region={region}",
        f"run.time.length={args.length}",
        f"run.execution.devices={args.devices}",
    ] + list(args.set)
    exp = clm_ml.load_experiment(folder, overrides)
    print(exp.summary())

    # A long or large run keeps only the records of the output file being filled in memory:
    # the results are read back from the netCDF files.
    exp.execute(return_outputs=False)
    files = sorted(exp.output_directory.glob("*_gridcell.nc"))
    print(f"\nwrote {exp.output_directory}/: {', '.join(f.name for f in files)}")
    if files:
        figure = plot(exp, files, exp.output_directory / f"{exp.name}_map.png")
        print(f"figure: {figure}")


def plot(exp, files, path: Path) -> Path:
    """Maps of the run-mean gridcell values (gridcells at their latitude and longitude)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr

    ds = xr.concat([xr.load_dataset(f) for f in files], dim="time").mean("time")
    # the gridcells on their latitude-longitude grid (land only; the rest stays blank)
    lat, lon = exp.domain.gridcells.latitude, exp.domain.gridcells.longitude
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    lats, row = np.unique(lat, return_inverse=True)
    lons, col = np.unique(lon, return_inverse=True)
    names = [v for v in MAPPED if v in ds]
    fig, axes = plt.subplots(
        (len(names) + 1) // 2, 2, figsize=(13, 3.5 * ((len(names) + 1) // 2)), squeeze=False
    )
    for ax, var in zip(axes.flat, names):
        v = ds[var]
        grid = np.full((len(lats), len(lons)), np.nan)
        grid[row, col] = v.values
        mesh = ax.pcolormesh(lons, lats, grid, shading="nearest", cmap="viridis")
        fig.colorbar(mesh, ax=ax, label=v.attrs.get("units", ""))
        ax.set_title(f"{var.replace('_', ' ')}: gridcell mean over the run", fontsize=9)
        ax.set_aspect("equal")
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


if __name__ == "__main__":
    main()
