"""
Setup and chunk inputs of grid runs (plan Phase 6b, steps 3 and 4).

The vectorized setup (:mod:`clm_ml.legacy.vector_setup`) builds every patch from a legacy
*prototype* setup plus per-patch formulas. Tower prototypes read the tower's forcing and CLM
history files; a grid run has no such files. So:

- **Prototypes, one per PFT**, are set up from small tower-format files written from one
  gridcell's gridded data (:func:`write_tower_files`): the first steps' meteorology as a
  tower forcing file and the first steps' vegetation, soil water and soil temperature as a
  CLM history file. They provide only the structure.
- **Every field that depends on the forcing and land data** is then computed per patch
  from its own gridcell's data (``vector_setup.setup_arrays(first=...)``): the forcing
  containers, the acclimation temperature and reference pressure, the soil temperature and
  water, the canopy-air and leaf states, and the first step's inputs. The list was found by
  perturbing each input of a tower one at a time (meteorology variables, soil temperature,
  soil water, leaf and stem area) and comparing all containers (Phase 6b step 3).
- The time steps read the gridded data chunk by chunk (:class:`GridChunkReader`,
  ``PatchBatch.reader``), and form each step's CLM history record inside the compiled chunk
  (``PatchBatch.land_fn``, :func:`clm_ml.grid_forcing.land_record`).

Test: ``tests/regression/test_grid_setup.py`` (the grid setup of a few gridcells equals the
legacy setup of towers whose files hold those gridcells' gridded data).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from clm_ml.forcing import chunk_run_steps

#: Soil layer counts (nlevsoi, nlevgrnd) of each layer structure (clm_varpar.clm_varpar_init);
#: needed before the first legacy setup sets clm_varpar
SOIL_LEVELS = {"clm5_0": (20, 25), "clm4_5": (10, 15)}

# ---------------------------------------------------------------------------------------
# Tower-format files from gridded data
# ---------------------------------------------------------------------------------------


def met_records(met: dict, steps) -> dict:
    """Rows ``steps`` of :meth:`GridInputs.met_at` output."""
    return {k: (tuple(x[steps] for x in v) if isinstance(v, tuple) else v[steps])
            for k, v in met.items()}  # fmt: skip


def write_tower_files(
    inputs, g: int, n_steps: int, out_dir: Path, levels: tuple, lai_patch=None
) -> tuple:
    """A tower forcing file and a CLM history file (30-min records, float64) holding gridcell
    ``g``'s gridded data for run steps ``0 ... n_steps - 1``: the derived meteorology at each
    step (``TBOT``, ``PSRF``, ``QBOT``, ``WIND``, ``FLDS``, ``FSDS_DIRECT``/``FSDS_DIFFUSE``,
    ``PRECTmms`` = rain), and ``ELAI``/``ESAI`` (of patch ``lai_patch``, default the
    gridcell's first), ``H2OSOI`` and ``TSOI`` as the grid run sees them at each step.
    The tower path then reads exactly the grid run's values (CO2 and snow excepted: tower
    files have neither; tower runs use 383 ppm and no snow)."""
    import netCDF4 as nc

    nlevsoi, nlevgrnd = levels
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    met = inputs.met_at(list(range(n_steps)))
    d = inputs.domain
    p = int(np.flatnonzero(d.patches.gridcell_index == g)[0]) if lai_patch is None else lai_patch
    dt_days = inputs.step_seconds / 86400.0
    origin = f"{inputs.start:%Y-%m-%d %H:%M:%S}"
    tower = out_dir / f"tower_g{g}.nc"
    with nc.Dataset(tower, "w") as ds:
        ds.createDimension("time", n_steps)
        ds.createDimension("lat", 1)
        ds.createDimension("lon", 1)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = f"days since {origin}"
        t[:] = np.arange(n_steps) * dt_days
        for name, value in (("LATIXY", d.gridcells.latitude[g]),
                            ("LONGXY", d.gridcells.longitude[g])):  # fmt: skip
            ds.createVariable(name, "f8", ("lat", "lon"))[:] = value
        columns = {
            "TBOT": met["tref"], "PSRF": met["pref"], "QBOT": met["qref"], "WIND": met["uref"],
            "FLDS": met["lwsky"], "FSDS_DIRECT": met["solad"][0] * 2.0,
            "FSDS_DIFFUSE": met["solai"][0] * 2.0, "PRECTmms": met["rain"],
        }  # fmt: skip
        for name, v in columns.items():
            ds.createVariable(name, "f8", ("time", "lat", "lon"))[:] = v[:, g][:, None, None]
    # vegetation and soil water at every step, as the compiled chunk forms them
    m1, m2, w1, w2 = inputs.vegetation_weights(0, n_steps)
    h, lb, f1, f2 = inputs.soil_window(0, n_steps, n_records=None)
    from clm_ml.grid_forcing import exposed_area

    elai, esai, h2o = [], [], []
    for k in range(n_steps):  # as land_record (no legacy import here: see setup_grid_batch)
        lai = w1[k] * inputs.lai[m1[k], p] + w2[k] * inputs.lai[m2[k], p]
        sai = w1[k] * inputs.sai[m1[k], p] + w2[k] * inputs.sai[m2[k], p]
        e_l, e_s = exposed_area(np.array(lai), np.array(sai))
        elai.append(float(e_l))
        esai.append(float(e_s))
        h2o.append(h[lb[k], g] * f1[k] + h[lb[k] + 1, g] * f2[k])
    tsoi = inputs.soil_at_first_step("TSOI")[g]
    hist = out_dir / f"history_g{g}_p{p}.nc"
    with nc.Dataset(hist, "w") as ds:
        ds.createDimension("time", n_steps)
        ds.createDimension("lndgrid", 1)
        ds.createDimension("levsoi", nlevsoi)
        ds.createDimension("levgrnd", nlevgrnd)
        ds.createDimension("hist_interval", 2)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = f"days since {origin}"
        t[:] = (np.arange(n_steps) + 1) * dt_days  # each slice ends at its step's end
        ds.createVariable("time_bounds", "f8", ("time", "hist_interval"))[:] = np.stack(
            [np.arange(n_steps) * dt_days, (np.arange(n_steps) + 1) * dt_days], 1
        )
        ds.createVariable("levgrnd", "f8", ("levgrnd",))[:] = np.arange(nlevgrnd)
        ds.createVariable("ELAI", "f8", ("time", "lndgrid"))[:] = np.array(elai)[:, None]
        ds.createVariable("ESAI", "f8", ("time", "lndgrid"))[:] = np.array(esai)[:, None]
        ds.createVariable("H2OSOI", "f8", ("time", "levsoi", "lndgrid"))[:] = np.array(h2o)[
            :, :, None
        ]
        ds.createVariable("TSOI", "f8", ("time", "levgrnd", "lndgrid"))[:] = np.repeat(
            tsoi[None, :, None], n_steps, axis=0
        )
    return tower, hist


def tower_spec(run, patch: int, tower_name: str, tower_file, history_file, n_steps=None) -> dict:
    """The legacy one-patch spec of domain patch ``patch`` of a grid run reading the given
    tower-format files (``Run._patch_spec`` for towers)."""
    from clm_ml.run import _legacy_tower_values

    namelist = {
        "tower_name": tower_name,
        "start_ymd": int(f"{run.start:%Y%m%d}"),
        "start_tod": run.start.hour * 3600 + run.start.minute * 60 + run.start.second,
        "stop_option": "nsteps",
        "stop_n": run.n_steps if n_steps is None else n_steps,
        "clm_start_ymd": int(f"{run.start:%Y%m%d}"),
        "clm_start_tod": run.start.hour * 3600 + run.start.minute * 60,
        "fin_tower": str(tower_file),
        "fin_tower_files": [str(tower_file)],
        "met_shortwave": "coszen",
        "met_precipitation": "constant",
        "fin_clm": str(history_file),
        "clm_phys": {"clm5_0": "CLM5_0", "clm4_5": "CLM4_5"}[run.config.soil.layer_structure],
        "fin_soil_adjust": " ",
        "fin_vegetation": " ",
        "fin_soil_water": " ",
        "nlev_soil_adjust": 0,
        "met_type": 3 if run.run_file.forcing.gridded.interpolation == "three_point" else 0,
        "dpai_min": run.config.canopy_layers.min_plant_area_per_layer,
        "pftcon_val": 0,
        "dtstep_seconds": run.step_seconds,
        "acclimation": run.config.physics.acclimation_temperature,
        "acclimation_days": run.config.physics.acclimation_days,
    }
    tower = _legacy_tower_values(run.domain, patch)
    tower["tower_ht"] = float(run.domain.patches.measurement_height[patch])
    tower["tower_time"] = run.step_seconds // 60  # records at the model step
    return {
        "namelist": namelist,
        "varctl": run.patch_fortran_codes(patch, run.run_file.forcing.gridded.interpolation),
        "tower_name": tower_name,
        "tower": tower,
    }


# ---------------------------------------------------------------------------------------
# Per-patch inputs
# ---------------------------------------------------------------------------------------


def tower_arrays(domain) -> dict:
    """The patches' own inputs in ``TowerDataMod`` form (``vector_setup.TOWER_INPUTS``)."""
    g, p, c = domain.gridcells, domain.patches, domain.columns
    gi, ci = p.gridcell_index, p.column_index

    def v(x):
        x = np.asarray(x, dtype=float)
        return np.where(np.isnan(x), -999.0, x)

    def pair(x):
        x = np.asarray(x, dtype=float)
        return np.where(np.isnan(x).any(axis=1, keepdims=True), -999.0, x)

    return {
        "tower_lat": v(g.latitude[gi]),
        "tower_lon": v(g.longitude[gi]),
        "tower_tex": [t or "" for t in np.asarray(c.texture_class, dtype=object)[ci]],
        "tower_sand": v(c.sand_percent[ci]),
        "tower_clay": v(c.clay_percent[ci]),
        "tower_organic": v(c.organic_matter[ci]),
        "tower_isoicol": np.asarray(c.color_class[ci], dtype=int),
        "tower_zbed": v(c.bedrock_depth[ci]),
        "tower_pft": np.asarray(p.pft_index, dtype=int),
        "tower_canht": v(p.canopy_height),
        "tower_root": v(p.root_biomass),
        "tower_pbeta_lai": pair(p.leaf_area_profile_beta),
        "tower_pbeta_sai": pair(p.stem_area_profile_beta),
    }


def atm_records(met: dict, current: bool):
    """``AtmForcingRecord`` arrays (leading axes of ``met``, then the legacy patch axis of
    length 1) as ``clm_ml.legacy.forcing.atm_record_current`` (``current``: wind resultant,
    CO2 from partial pressure) or ``atm_record_next`` builds them, with the same arithmetic."""
    from clm_src_main.clm_varpar import inir, ivis, numrad
    from multilayer_canopy.MLGetAtmForcingMod import AtmForcingRecord

    f = lambda x: np.asarray(x, dtype=float)  # noqa: E731
    if current:
        uref = np.sqrt(f(met["uref"]) ** 2 + 0.0**2)
        co2 = f(met["pco2"]) / f(met["pref"]) * 1.0e6
    else:
        uref, co2 = f(met["uref"]), f(met["co2_ppm"])
    shape = uref.shape

    def bands(pair):
        out = np.zeros(shape + (1, numrad + 1))
        out[..., 0, ivis], out[..., 0, inir] = f(pair[0]), f(pair[1])
        return out

    one = lambda x: f(x)[..., None]  # noqa: E731
    return AtmForcingRecord(
        uref=one(uref), tref=one(met["tref"]), qref=one(met["qref"]), pref=one(met["pref"]),
        co2ref=one(co2), swskyb=bands(met["solad"]), swskyd=bands(met["solai"]),
        lwsky=one(met["lwsky"]),
    )  # fmt: skip


def tower_met(met: dict):
    """``TowerMet`` with array fields from a :meth:`GridInputs.met_at` dict."""
    from offline_driver.TowerMetMod import TowerMet

    return TowerMet(**{k: (tuple(np.asarray(x, float) for x in v) if isinstance(v, tuple)
                           else np.asarray(v, float)) for k, v in met.items()})  # fmt: skip


def first_step_inputs(inputs, run) -> dict:
    """Each patch's first-step data for ``vector_setup.setup_arrays(first=...)``."""
    from clm_ml.grid_forcing import land_record_np

    _, nlevgrnd = SOIL_LEVELS[run.config.soil.layer_structure]

    d = inputs.domain
    gi = d.patches.gridcell_index
    rs = chunk_run_steps(run.n_steps, 0, 1)  # [halo (= step 0), 0, next]
    met = inputs.met_at(rs)
    per_patch = {k: (tuple(x[:, gi] for x in v) if isinstance(v, tuple) else v[:, gi])
                 for k, v in met.items()}  # fmt: skip
    m1, m2, w1, w2 = inputs.vegetation_weights(0, 1)
    lai = w1[0] * inputs.lai[m1[0]] + w2[0] * inputs.lai[m2[0]]
    sai = w1[0] * inputs.sai[m1[0]] + w2[0] * inputs.sai[m2[0]]
    h, lb, f1, f2 = inputs.soil_window(0, 1)
    h2o = (h[lb[0]] * f1[0] + h[lb[0] + 1] * f2[0])[gi]
    return {
        "zref": d.patches.measurement_height,
        "met": tower_met(met_records(per_patch, 1)),
        "forcing_bef": atm_records(met_records(per_patch, 0), current=True),
        "forcing_next": atm_records(met_records(per_patch, 2), current=False),
        "clm_record": land_record_np(lai, sai, h2o, nlevgrnd),
        "tsoi": inputs.soil_at_first_step("TSOI")[gi],
        "h2osoi": inputs.soil_at_first_step("H2OSOI")[gi],
    }


# ---------------------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------------------


def grid_inputs(run):
    """The :class:`clm_ml.grid_forcing.GridInputs` of a grid run."""
    from clm_ml.grid_forcing import GridInputs

    return GridInputs(
        run.domain, run.path(run.run_file.domain.grid.surface_data), run.run_file.forcing.gridded,
        run.start, run.step_seconds, run.n_steps, resolve=run.path,
    )  # fmt: skip


def setup_grid_batch(spec: dict, run=None, inputs=None, workdir=None):
    """The ``PatchBatch`` of a grid run (domain order, not yet laid out for devices) from
    the worker spec (``spec["grid"]``: the run file and overrides)."""
    from clm_ml.legacy import vector_setup
    from clm_ml.run import load_run

    if run is None:
        run = load_run(spec["grid"]["run_file"], spec["grid"].get("overrides", ()))
    if inputs is None:
        inputs = grid_inputs(run)
    d = run.domain
    tmp = Path(workdir or tempfile.mkdtemp(prefix="clm_ml_grid_"))
    # prototypes: one per PFT, from gridcell 0's data (structure only)
    levels = SOIL_LEVELS[run.config.soil.layer_structure]
    tower_file, history_file = write_tower_files(inputs, 0, 2, tmp, levels)
    pft = d.patches.pft_index
    proto_specs = {}
    for k in sorted(set(pft.tolist())):
        patch = int(np.flatnonzero(pft == k)[0])
        ps = tower_spec(run, patch, f"PROTO_PFT{k}", tower_file, history_file)
        proto_specs[k] = {**spec, **ps}
    layers = run.config.canopy_layers  # constant layer counts (decision 28)
    within = np.full(d.n_patches, layers.n_layers_within, dtype=int)
    above = np.full(d.n_patches, layers.n_layers_above, dtype=int)
    names = [f"{d.gridcells.names[g]}#{k}" for g, k in zip(d.patches.gridcell_index, pft)]
    # the first-step data are built after the prototypes' setups (they use legacy records,
    # whose modules must not be imported before the first setup sets the soil layer counts)
    batch = vector_setup.setup_arrays(
        proto_specs, [int(k) for k in pft], tower_arrays(d), within, above,
        d.patches.gridcell_index, d.patches.weight, d.n_gridcells, names, host=True,
        first=lambda: first_step_inputs(inputs, run),
    )  # fmt: skip
    return batch, inputs


class GridChunkReader:
    """``PatchBatch.reader`` of a grid run: a chunk's ``(xg, xp)`` for the batch's slots
    (padding gridcells read the last real gridcell's data, padding patches the patch they
    copy). ``xg``: the step times ``(n,)`` and the meteorology per gridcell slot; ``xp``:
    ``(xw, land)``, per-step vegetation and soil-water weights and the chunk's monthly
    records (:func:`clm_ml.grid_forcing.land_record`)."""

    def __init__(self, inputs, batch):
        self.inputs, self.batch = inputs, batch
        self.src_g = np.minimum(np.arange(batch.n_gridcells), batch.n_real_gridcells - 1)
        self.src_p = np.asarray(batch.source)

    def __call__(self, first_step: int, n_steps: int):
        import jax

        from clm_src_utils.clm_time_manager import step_time

        inp, run = self.inputs, self.batch.run
        rs = chunk_run_steps(run.ntimes, first_step, n_steps)
        met = inp.met_at(rs)
        met = {k: (tuple(x[:, self.src_g] for x in v) if isinstance(v, tuple) else v[:, self.src_g])
               for k, v in met.items()}  # fmt: skip
        n = n_steps
        cur = met_records(met, slice(1, n + 1))
        bef = atm_records(met_records(met, slice(0, n)), current=True)
        nxt = atm_records(met_records(met, slice(2, n + 2)), current=False)
        start = inp.start
        ymd, tod = int(f"{start:%Y%m%d}"), start.hour * 3600 + start.minute * 60 + start.second
        times = [
            step_time(ymd, tod, inp.step_seconds, k + 1) for k in range(first_step, first_step + n)
        ]
        now = type(times[0])(*(np.array(col) for col in zip(*times)))
        xg = (now, tower_met(cur), bef, nxt)
        m1, m2, w1, w2 = inp.vegetation_weights(first_step, n)
        h, lb, f1, f2 = inp.soil_window(first_step, n)
        xw = (m1, m2, w1, w2, lb, f1, f2)
        land = (inp.lai[:, self.src_p], inp.sai[:, self.src_p], h[:, self.src_g])
        # host arrays: the run places them on the devices, split by patch (decision 33)
        return jax.tree.map(np.asarray, xg), jax.tree.map(np.asarray, (xw, land))


def land_fn(xw, land, gidx):
    """``PatchBatch.land_fn`` of grid runs."""
    from clm_ml.grid_forcing import land_record
    from clm_src_main import clm_varpar

    return land_record(xw, land, gidx, clm_varpar.nlevgrnd)
