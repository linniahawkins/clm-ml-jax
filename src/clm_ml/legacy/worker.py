"""
Run the legacy (Fortran-translated) driver from a ``clm_ml`` run specification.

``Run.execute()`` starts this module in a fresh Python process::

    python -m clm_ml.legacy.worker SPEC.json OUT.npz

A fresh process is required because the legacy code keeps model state, options and
parameter tables in module globals, and most modules copy those globals when they are
first imported. So the order here matters:

1. enable float64,
2. set every ``MLclm_varctl`` option,
3. build the PFT parameter tables from ``Params`` and install them (replacing both the
   tables and the ``Init`` functions that ``initialize2`` calls to rebuild them),
4. write the tower into ``TowerDataMod``,
5. only then run the model: ``offline_executable.main.setup_run`` and ``run_streams`` (the
   same physics as ``clm-ml-offline``), accumulating the output streams inside each
   compiled chunk and saving their records. With ``mode: legacy_ascii`` it instead runs the
   legacy driver to write the Fortran-format .out files (validation only).

The spec is produced by :meth:`clm_ml.run.Run.legacy_spec`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

#: Fields of ``mlcanopy_inst`` read by ``CLMml_driver.output()`` (everything that reaches
#: the legacy .out files). Recorded every time step as ``mlcanopy.<field>``.
MLCANOPY_FIELDS = (
    "albcan_canopy", "anet_leaf", "apar_leaf", "beta_canopy", "btran_soil", "dpai_profile",
    "dz_profile", "eair_profile", "etair_profile", "evveg_canopy", "fracminlwp_canopy",
    "fracsun_profile", "gac_profile", "gppveg_canopy", "gppvegsha_canopy", "gppvegsun_canopy",
    "gs_leaf", "gsoi_soil", "gsveg_canopy", "gsvegsha_canopy", "gsvegsun_canopy", "lai_canopy",
    "laisha_canopy", "laisun_canopy", "lhflx_canopy", "lhleaf_leaf", "lhsoi_soil", "lhveg_canopy",
    "lhvegsha_canopy", "lhvegsun_canopy", "lsc_profile", "lwdwn_profile", "lwp_hist_leaf",
    "lwp_mean_profile", "lwup_canopy", "lwupw_profile", "mflx_profile", "nbot_canopy",
    "ncan_canopy", "ntop_canopy", "pref_forcing", "psis_soil", "rhomol_forcing", "rnet_canopy",
    "rnleaf_leaf", "rnsoi_soil", "sai_canopy", "sai_floor_canopy", "shair_profile", "shflx_canopy", "shleaf_leaf",
    "shsoi_soil", "shveg_canopy", "shvegsha_canopy", "shvegsun_canopy", "solar_zen_forcing",
    "stflx_air_canopy", "stflx_veg_canopy", "swbeam_profile", "swdwn_profile", "swskyb_forcing",
    "swskyd_forcing", "swupw_profile", "swveg_canopy", "swvegsha_canopy", "swvegsun_canopy",
    "tair_profile", "taveg_canopy", "tavegsha_canopy", "tavegsun_canopy", "tleaf_hist_leaf",
    "tlveg_canopy", "tlvegsha_canopy", "tlvegsun_canopy", "tref_forcing", "trveg_canopy",
    "ustar_canopy", "vcmax25_leaf", "vcmax25sha_canopy", "vcmax25sun_canopy",
    "vcmax25veg_canopy", "wind_profile", "windveg_canopy", "windvegsha_canopy",
    "windvegsun_canopy", "zs_profile", "zw_profile",
)  # fmt: skip
TEMPERATURE_FIELDS = ("t_soisno_col",)


def prepare_process(spec: dict) -> None:
    """Steps 1-4 above: configure this (fresh) process for ``spec``. The legacy modules keep
    tables and the tower table in module state; it can be prepared again for another model
    of the same tower (each model holds its own options and parameters after setup)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    # 1b. Model calendar (run.yaml time.calendar), read by the time manager at every date
    from clm_src_utils import clm_time_manager

    clm_time_manager.calkindflag = spec.get("calendar", "GREGORIAN")

    # 2. Options -- before any physics module copies them at import.
    from multilayer_canopy import MLclm_varctl

    for name, value in spec["varctl"].items():
        setattr(MLclm_varctl, name, value)
    MLclm_varctl.pftcon_val = 0  # parameter values come from Params below, not the switch

    # 3. PFT parameter tables.
    from clm_ml.legacy.tables import params_to_legacy
    from clm_ml.params import Params
    from clm_src_main import pftconMod
    from multilayer_canopy import MLpftconMod

    params = Params(
        values={k: np.array(v, dtype=float) for k, v in spec["params"].items()},
        source=spec["params_source"],
    )
    base = {}
    for table_name, table in (("pftcon", pftconMod.pftcon), ("MLpftcon", MLpftconMod.MLpftcon)):
        for field in table._fields:
            base[f"{table_name}.{field}"] = np.asarray(getattr(table, field))
    tables = params_to_legacy(params, base)
    new_pftcon = pftconMod.pftcon._replace(
        **{f: jnp.asarray(tables[f"pftcon.{f}"]) for f in pftconMod.pftcon._fields}
    )
    new_mlpftcon = MLpftconMod.MLpftcon._replace(
        **{f: jnp.asarray(tables[f"MLpftcon.{f}"]) for f in MLpftconMod.MLpftcon._fields}
    )
    pftconMod.pftcon, pftconMod.Init = new_pftcon, (lambda: new_pftcon)
    MLpftconMod.MLpftcon, MLpftconMod.Init = new_mlpftcon, (lambda: new_mlpftcon)

    # 4. Tower table.
    _install_tower(spec["tower_name"], spec["tower"])


def setup_legacy_run(spec: dict):
    """``offline_executable.main.setup_run`` for ``spec`` (after :func:`prepare_process`)."""
    import offline_executable.main as main_mod

    with tempfile.TemporaryDirectory() as tmp:
        namelist = _write_namelist(spec["namelist"], Path(tmp), None)
        return main_mod.setup_run(main_mod.read_namelist(str(namelist)), _options(spec))


def setup_batch(spec: dict):
    """Every patch of ``spec`` as an ``offline_executable.main.PatchBatch``: one legacy
    setup per (forcing key, PFT) prototype, every patch's own fields computed for all patches
    at once (``clm_ml.legacy.vector_setup``, bit for bit equal to setting each patch up,
    :func:`setup_batch_legacy`), the first step's physics batched; then laid out for the
    devices and C3/C4 groups (Phase 6 steps 3-6). Grid runs (``spec["grid"]``, Phase 6b):
    prototypes per PFT from the gridded data, the chunks read gridded data
    (``clm_ml.legacy.grid_setup``)."""
    from clm_ml.legacy import vector_setup

    if spec.get("grid"):
        from clm_ml.legacy import grid_setup

        domain_batch, inputs = grid_setup.setup_grid_batch(spec)
        batch = laid_out(domain_batch, spec["batch"])
        batch.reader = grid_setup.GridChunkReader(inputs, batch)
        batch.land_fn = grid_setup.land_fn
        return batch
    return laid_out(vector_setup.setup_batch(spec), spec["batch"])


def laid_out(domain_batch, batch: dict):
    """The domain-order batch laid out for ``batch["devices"]`` devices and C3/C4 groups
    (block layout), or padded to ``patch_count``/``gridcell_count``."""
    grids = domain_batch.grids
    devices = int(batch.get("devices") or 1)
    mixed = len({g.c3 for g in grids}) > 1
    if devices > 1 or mixed:
        # step 5: the same block of patches and gridcells on every device; C3 and C4 patches
        # always in contiguous groups (step 4), also on one device
        if batch.get("patch_count") or batch.get("gridcell_count"):
            raise ValueError(
                "execution.patch_count/gridcell_count: not supported with several devices or "
                "with C3 and C4 patches in one run (the blocks are padded automatically)"
            )
        return domain_batch.blocked(devices)
    return domain_batch.padded(batch.get("patch_count"), batch.get("gridcell_count"))


def setup_batch_legacy(spec: dict):
    """Every patch of ``spec`` (``spec["patches"]``) set up as a one-patch legacy run, one
    after the other in this process (each installs its tower entry and tables first, and runs
    its first-step initialization before the next setup; the first step's physics is then
    batched over the patches), as an ``offline_executable.main.PatchBatch``. The validation
    reference of :func:`setup_batch` (``tests/regression/test_vector_setup.py``)."""
    import offline_executable.main as main_mod

    runs, grids, inits = [], [], []
    for patch in spec["patches"]:
        patch_spec = {**spec, **patch}
        prepare_process(patch_spec)
        runs.append(setup_legacy_run(patch_spec))
        # before the next patch's setup replaces the module state they read: the patch's
        # GridInfo and its first-step initialization (the step's physics is batched later)
        grids.append(main_mod.patch_grid(runs[-1]))
        inits.append(main_mod.PatchBatch.initialize_patch(runs[-1], grids[-1]))
    batch = spec["batch"]
    domain_batch = main_mod.PatchBatch(
        runs,
        np.asarray(batch["gridcell_index"], dtype=int),
        np.asarray(batch["weight"], dtype=float),
        int(batch["n_gridcells"]),
        [patch["tower_name"] for patch in spec["patches"]],
        grids,
        inits=inits,
    )
    return laid_out(domain_batch, batch)


def _options(spec: dict):
    """The multilayer canopy options of ``spec`` (model.yaml), passed to the physics."""
    from multilayer_canopy.MLclm_varctl import MLOptions

    return MLOptions.from_codes(spec["varctl"])


def run_spec(spec: dict, out_file: Path) -> None:
    if spec.get("mode") == "legacy_ascii":
        # Validation output only (one patch): the step-by-step legacy driver writing the six
        # .out files.
        prepare_process(spec)
        import offline_executable.main as main_mod

        with tempfile.TemporaryDirectory() as tmp:
            namelist = _write_namelist(spec["namelist"], Path(tmp), spec["legacy_ascii_dir"])
            sys.argv = ["clm-ml-offline", str(namelist)]
            main_mod.main(_options(spec))
        return

    # 5. Set up every patch, then run them together.
    import time

    t0 = time.perf_counter()
    batch = setup_batch(spec)
    if spec.get("grid"):
        print(f"timing: setup {time.perf_counter() - t0:.1f} s ({batch.n_real} patches, "
              f"{batch.n_real_gridcells} gridcells, layout {batch.layout})", flush=True)  # fmt: skip
    import datetime as dt

    import offline_executable.main as main_mod
    from clm_ml.output_streams import StreamSpec
    from clm_ml.outputs import StreamFileWriter
    from clm_src_main.clm_varpar import nlevgrnd

    streams = [
        StreamSpec(s["name"], s["frequency"], s["statistic"], tuple(s["variables"]))
        for s in spec["streams"]
    ]
    levels = [tuple(s["level"]) for s in spec["streams"]]
    # Each stream's values: its variables, plus the gridcell sums formed in the chunk
    keys = [
        tuple(stream.variables)
        + tuple(f"gridcell:{n}" for n in main_mod.gridcell_variables(stream.variables, level))
        for stream, level in zip(streams, levels)
    ]
    meta = dict(spec["meta"])
    if spec.get("grid"):  # one soil layer structure for every patch
        z0 = np.asarray(batch.runs[0].insts.site.z_col)[:, 1 : nlevgrnd + 1]
        meta["soil_depth"] = np.repeat(z0, batch.n_real, axis=0)
    else:
        meta["soil_depth"] = np.concatenate(  # (patch, soil layer) node depths
            [np.asarray(r.insts.site.z_col)[:, 1 : nlevgrnd + 1] for r in batch.runs]
        )
    writers = None
    if spec.get("output_dir"):
        start = dt.datetime.fromisoformat(spec["run_start"])
        writers = [
            StreamFileWriter(
                _FileOptions(s["level"], s["file_per"], s["name"]),
                stream_keys,
                meta,
                Path(spec["output_dir"]),
                spec.get("file_prefix", spec["tower_name"]),
                start,
            )
            for s, stream_keys in zip(spec["streams"], keys)
        ]
    keep = bool(spec.get("return_records", True))
    collected: list[list] = [[] for _ in streams]

    def on_records(i, records):
        if writers is not None:
            writers[i].add(records)  # files are written as their periods complete
        if keep:
            collected[i].extend(records)

    def save_extra():
        # the output files in progress (their records are written first); no output history:
        # a restart holds what the next run needs, not the records so far (review R6), so a
        # resumed run returns the records from the restart on
        return {"writers": [w.state() for w in writers] if writers is not None else None}

    def restore_extra(saved):
        if saved is None:
            return
        if writers is not None and saved.get("writers") is not None:
            for w, ws in zip(writers, saved["writers"]):
                w.restore(ws)

    restart = main_mod.read_restart(Path(spec["restart_in"])) if spec.get("restart_in") else None
    every = spec.get("restart_every")
    main_mod.run_streams(
        batch,
        streams,
        spec["max_layers"],
        on_records,
        restart=restart,
        restart_path=Path(spec["restart_out"]) if spec.get("restart_out") else None,
        restart_every=tuple(every) if every else None,
        save_extra=save_extra,
        restore_extra=restore_extra,
        levels=levels,
        on_patch_error=spec.get("on_patch_error", "stop"),
        max_failed_patch_fraction=spec.get("max_failed_patch_fraction", 0.01),
        ignore_checks=tuple(spec.get("ignore_checks", ())),
        chunk_steps=spec.get("chunk_steps"),
    )
    if writers is not None:
        for w in writers:
            w.close()
    if spec.get("grid"):
        import jax

        stats = [d.memory_stats() or {} for d in jax.local_devices()]
        peak = [round(st.get("peak_bytes_in_use", 0) / 1e9, 2) for st in stats]
        print(f"timing: peak device memory GB {peak}", flush=True)

    arrays = {"static.soil_depth": meta["soil_depth"]}
    if keep:
        from clm_ml.outputs import records_to_arrays

        for stream, stream_keys, records in zip(streams, keys, collected):
            if not records:
                raise RuntimeError(f"output stream {stream.name} produced no records")
            for key, value in records_to_arrays(records, stream_keys).items():
                arrays[f"{stream.name}/{key}"] = value
    np.savez_compressed(out_file, **arrays)


class _FileOptions:
    """The parts of a run.yaml stream the file writer needs."""

    def __init__(self, level, file_per, name):
        self.level, self.file_per, self.name = tuple(level), file_per, name


def _install_tower(name: str, values: dict) -> None:
    """Write one tower's values into ``TowerDataMod`` (appending it if the name is new)."""
    import jax.numpy as jnp

    from offline_driver import TowerDataMod as T

    ids = [str(t) for t in T.tower_id]
    if name in ids:
        index = ids.index(name)
    else:
        index = len(ids)
        T.tower_id = list(T.tower_id) + [name]
        T.tower_tex = list(T.tower_tex) + [""]
        for array_name in values:
            if array_name in ("tower_id", "tower_tex"):
                continue
            array = getattr(T, array_name)
            pad = jnp.zeros((1,) + array.shape[1:], dtype=array.dtype)
            setattr(T, array_name, jnp.concatenate([array, pad]))
        T.ntower = index
    for array_name, value in values.items():
        if array_name == "tower_tex":
            T.tower_tex[index] = value
        else:
            array = getattr(T, array_name)
            setattr(T, array_name, array.at[index].set(jnp.asarray(value, dtype=array.dtype)))


def _write_namelist(values: dict, workdir: Path, legacy_ascii_dir) -> Path:
    import f90nml

    group = dict(values)
    out_dir = Path(legacy_ascii_dir) if legacy_ascii_dir else workdir / "discarded_ascii"
    out_dir.mkdir(parents=True, exist_ok=True)
    group["dirout"] = str(out_dir) + "/"
    path = workdir / "run.nl"
    f90nml.Namelist({"clmML_inparm": group}).write(str(path), force=True)
    return path


def main() -> None:
    spec_file, out_file = Path(sys.argv[1]), Path(sys.argv[2])
    run_spec(json.loads(spec_file.read_text()), out_file)


if __name__ == "__main__":
    main()
