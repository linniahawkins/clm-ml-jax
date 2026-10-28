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
    "rnleaf_leaf", "rnsoi_soil", "sai_canopy", "shair_profile", "shflx_canopy", "shleaf_leaf",
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


def run_spec(spec: dict, out_file: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

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

    # 5. Run.
    import offline_executable.main as main_mod

    with tempfile.TemporaryDirectory() as tmp:
        if spec.get("mode") == "legacy_ascii":
            # Validation output only: the step-by-step legacy driver writing the six .out files.
            namelist = _write_namelist(spec["namelist"], Path(tmp), spec["legacy_ascii_dir"])
            sys.argv = ["clm-ml-offline", str(namelist)]
            main_mod.main()
            return
        namelist = _write_namelist(spec["namelist"], Path(tmp), None)
        run = main_mod.setup_run(main_mod.read_namelist(str(namelist)))

    import datetime as dt

    from clm_ml.output_streams import StreamSpec
    from clm_ml.outputs import StreamFileWriter
    from clm_src_main import ColumnType
    from clm_src_main.clm_varpar import nlevgrnd

    streams = [
        StreamSpec(s["name"], s["frequency"], s["statistic"], tuple(s["variables"]))
        for s in spec["streams"]
    ]
    meta = dict(spec["meta"])
    meta["soil_depth"] = np.asarray(ColumnType.col.z)[:, 1 : nlevgrnd + 1]
    writers = None
    if spec.get("output_dir"):
        start = dt.datetime.fromisoformat(spec["run_start"])
        writers = [
            StreamFileWriter(
                _FileOptions(s["level"], s["file_per"], s["name"]),
                stream.variables,
                meta,
                Path(spec["output_dir"]),
                spec["tower_name"],
                start,
            )
            for s, stream in zip(spec["streams"], streams)
        ]
    keep = bool(spec.get("return_records", True))
    collected: list[list] = [[] for _ in streams]

    def on_records(i, records):
        if writers is not None:
            writers[i].add(records)  # files are written as their periods complete
        if keep:
            collected[i].extend(records)

    main_mod.run_streams(run, streams, spec["max_layers"], on_records)
    if writers is not None:
        for w in writers:
            w.close()

    arrays = {"static.soil_depth": meta["soil_depth"]}
    if keep:
        from clm_ml.outputs import records_to_arrays

        for stream, records in zip(streams, collected):
            if not records:
                raise RuntimeError(f"output stream {stream.name} produced no records")
            for key, value in records_to_arrays(records, stream.variables).items():
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
