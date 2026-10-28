"""
Routine-level reference: real inputs and outputs of each physics routine.

Phase 3c merges the "normal" and "traceable" versions of several routines into one
(docs/design/numerics_and_regression.md §1). This records, from the code before the merge,
the inputs and outputs of each such routine at three moments of the CHATS7 reference day.
The merged routines are then replayed on the same inputs and must reproduce the outputs to
round-off.

    python tests/regression/process_io.py --record    # writes reference/process_io.pkl.gz
    python tests/regression/process_io.py --replay OUT.json

Both run the model in this process (it initializes module-level tables), so they are
started in a fresh process by the test.
"""

from __future__ import annotations

import argparse
import gzip
import inspect
import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parents[1] / "src"
PROCESS_IO_FILE = HERE / "reference" / "process_io.pkl.gz"
NAMELIST = SRC_DIR / "offline_executable" / "nl.CHATS7.50steps"

ROUTINES = (
    "GetAtmForcing",
    "SolarRadiation",
    "LongwaveRadiation",
    "CanopyTurbulence",
    "LeafPhotosynthesis",
    "FluxProfileSolution",
    "RungeKuttaUpdate",
)
# CLM steps (1-based) to record: dusk transition, night, midday (PST = UTC - 8).
RECORD_STEPS = (6, 14, 40)
CALLS_PER_STEP = 2  # first calls of each routine in the step (covers sun and shade leaves)


def _to_numpy(tree):
    import jax

    return jax.tree_util.tree_map(lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, tree)


def _to_jax(tree):
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, tree
    )


def _run_model(n_steps: int, hook) -> None:
    """Run the CHATS7 case for ``n_steps`` with ``hook(F)`` called after initialization,
    where ``F`` is the MLCanopyFluxesMod module."""
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(HERE.parents[1]))
    import offline_executable.main as main_mod
    from tests.regression.record_run import write_portable_namelist

    original_initialize = main_mod.InitializeRealize

    def initialize(bounds, *args):
        original_initialize(bounds, *args)
        import multilayer_canopy.MLCanopyFluxesMod as F

        hook(F)

    main_mod.InitializeRealize = initialize
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "out").mkdir()
        nml = write_portable_namelist(NAMELIST, tmp / "nl", tmp / "out")
        text = nml.read_text().replace("stop_n = 50", f"stop_n = {n_steps}")
        # Step by step (chunk_steps = 0): the routines see concrete values to record, not the
        # traced values of a compiled chunk.
        body = text.rstrip()
        assert body.endswith("/"), "namelist group should end with '/'"
        text = body[:-1] + "    chunk_steps = 0\n/\n"
        nml.write_text(text)
        sys.argv = ["clm-ml-offline", str(nml)]
        main_mod.main()


def record() -> None:
    records = {name: [] for name in ROUTINES}
    state = {"step": 0, "counts": {}}

    def hook(F):
        original_mlcanopy = F.MLCanopyFluxes

        def counting(*args, **kwargs):
            state["step"] += 1
            state["counts"] = {}
            return original_mlcanopy(*args, **kwargs)

        F.MLCanopyFluxes = counting
        import clm_src_main.clm_driver as D

        D.MLCanopyFluxes = counting
        for name in ROUTINES:
            original = getattr(F, name)

            def make(name, original):
                def wrapped(*args, **kwargs):
                    out = original(*args, **kwargs)
                    n = state["counts"].get(name, 0)
                    state["counts"][name] = n + 1
                    if state["step"] in RECORD_STEPS and n < CALLS_PER_STEP:
                        records[name].append(
                            {
                                "step": state["step"],
                                "args": _to_numpy(args),
                                "kwargs": _to_numpy(kwargs),
                                "output": _to_numpy(out),
                            }
                        )
                    return out

                return wrapped

            setattr(F, name, make(name, original))

    _run_model(max(RECORD_STEPS), hook)
    PROCESS_IO_FILE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(PROCESS_IO_FILE, "wb") as fh:
        pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
    sizes = {k: len(v) for k, v in records.items()}
    print(f"wrote {PROCESS_IO_FILE} ({PROCESS_IO_FILE.stat().st_size / 1e6:.1f} MB): {sizes}")


def replay(out_json: Path) -> None:
    """Re-run every recorded call with the current routines; report the largest relative
    difference per routine and call. The recording (made before Phase 3c) passed
    ``grid=None``; the merged routines require a ``GridInfo``, built from the recorded state."""
    report = {}

    def hook(F):
        report["module"] = F

    # Initialize first: unpickling imports model modules, and most of them copy
    # MLclm_varctl options (e.g. met_type) at import, so they must be imported after
    # the run configuration is set (see CLAUDE.md, "Gotchas").
    _run_model(1, hook)  # initializes tables (PFT parameters, lookup tables, filters)
    F = report.pop("module")
    with gzip.open(PROCESS_IO_FILE, "rb") as fh:
        records = _Unpickler(fh).load()
    for name, calls in records.items():
        results = []
        for call in calls:
            kwargs = dict(call["kwargs"])
            kwargs.pop("g1_MED_jax", None)  # replaced by params (Phase 3d)
            kwargs.pop("_o2ref_py", None)  # O2 is a runtime kernel argument (Phase 4)
            if kwargs.get("grid", "absent") is None:
                kwargs.update(_grid_kwargs(name, call["args"]))
            if "opts" in inspect.signature(getattr(F, name)).parameters:
                kwargs.setdefault("opts", _chats7_options())  # explicit options (Phase 5a)
            out = getattr(F, name)(*_to_jax(call["args"]), **_to_jax(kwargs))
            results.append(
                {"step": call["step"], "max_rel_diff": _max_rel_diff(call["output"], out)}
            )
        report[name] = results
    Path(out_json).write_text(json.dumps(report, indent=1))


class _Unpickler(pickle.Unpickler):
    """Reads recordings made before Phase 5b, whose ``GridInfo`` also held the layer counts
    and the C3/C4 flag ``(p, ncan, ntop, nbot, c3)``: those are data now."""

    def find_class(self, module, name):
        cls = super().find_class(module, name)
        if name == "CanopyParams":  # recorded before c3_photosynthesis was a parameter

            class OldCanopyParams(cls):
                def __new__(klass, *fields):
                    if len(fields) == len(cls._fields) - 1:
                        values = dict(zip([f for f in cls._fields if f != "c3_photosynthesis"],
                                          fields))  # fmt: skip
                        return cls(**values, c3_photosynthesis=1.0)  # CHATS7: C3
                    return cls(*fields)

            return OldCanopyParams
        if name != "GridInfo":
            return cls

        class OldGridInfo(cls):
            def __new__(klass, *fields):
                return cls(p=fields[0])

        return OldGridInfo


#: The multilayer canopy options of the recorded CHATS7 run (the Fortran defaults with the
#: namelist's three-point forcing interpolation); the recording predates explicit options.
def _chats7_options():
    from multilayer_canopy.MLclm_varctl import DEFAULT_OPTIONS

    return DEFAULT_OPTIONS._replace(met_type=3)


# Routines that take the patch's parameters (Phase 3d); the recording predates them.
TAKES_PARAMS = ("SolarRadiation", "LongwaveRadiation", "LeafPhotosynthesis")


def _grid_kwargs(name: str, args) -> dict:
    """``GridInfo`` and the patch's ``CanopyParams`` from the recorded multilayer canopy
    state among ``args`` (single-patch runs)."""
    from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type
    from multilayer_canopy.MLCanopyParamsMod import patch_setup

    inst = next(a for a in args if isinstance(a, mlcanopy_type))
    p = 0
    grid, params = patch_setup(p, inst)
    kwargs = {"grid": grid}
    if name in TAKES_PARAMS:
        kwargs["params"] = params
    return kwargs


def _max_rel_diff(expected, actual) -> dict:
    """Largest difference per field, relative to the field's magnitude (fields that differ)."""
    diffs = {}
    for field in expected._fields:
        x = np.asarray(getattr(expected, field), dtype=float)
        y = np.asarray(getattr(actual, field), dtype=float)
        if x.shape != y.shape:
            diffs[field] = float("inf")
            continue
        valid = np.isfinite(x) & (np.abs(x) < 1e30)
        if not valid.any():
            continue
        scale = float(np.max(np.abs(x[valid]))) or 1.0
        d = float(np.max(np.abs(x[valid] - y[valid]))) / scale
        if d > 0:
            diffs[field] = d
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--replay", type=Path, metavar="OUT_JSON")
    args = parser.parse_args()
    if args.record:
        record()
    elif args.replay:
        replay(args.replay)
    else:
        parser.error("pass --record or --replay OUT_JSON")


if __name__ == "__main__":
    main()
