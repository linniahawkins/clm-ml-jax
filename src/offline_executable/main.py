"""
main.py  —  JAX equivalent of prgm.exe (offline_executable).

Run with:
    python main.py < nl.CHATS7.05.2007
or:
    python main.py nl.CHATS7.05.2007

Mirrors the Fortran entry point in offline_driver/CLMml.F90 and
the namelist reader in offline_driver/controlMod.F90.
"""

import dataclasses
import datetime
import os
import sys
import time
from pathlib import Path
from typing import Optional

import f90nml  # pip install f90nml
import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit floats in JAX before any module-level array creation.
jax.config.update("jax_enable_x64", True)

# Persistent JIT compilation cache — avoids 290s recompile on subsequent runs.
# Cache dir can be overridden via JAX_COMPILATION_CACHE_DIR env var.
_jax_cache_dir = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.expanduser("~/.cache/jax_compile_cache"),
)
os.makedirs(_jax_cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _jax_cache_dir)
# Every compiled program is cached, small ones included: a run compiles ~500 small programs
# (the eagerly run first step), ~35 s that a new process would otherwise repeat. With the
# filelock package installed, least recently used entries are removed beyond 2 GB.
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
try:
    import filelock  # noqa: F401

    jax.config.update("jax_compilation_cache_max_size", 2 * 1024**3)
except ImportError:
    pass

from clm_ml.legacy.state import instances_with_state, state_from_instances  # noqa: E402
from clm_src_cpl.lnd_comp_nuopc import InitializeRealize, ModelAdvance  # noqa: E402
from clm_src_main.decompMod import bounds_type  # noqa: E402
from multilayer_canopy.MLCanopyFluxesType import BALANCE_CHECKS, check_balance  # noqa: E402
from multilayer_canopy.MLclm_varctl import DEFAULT_OPTIONS, MLOptions  # noqa: E402
from multilayer_canopy.MLMathToolsMod import is_concrete  # noqa: E402
from offline_driver import controlMod  # noqa: E402

# Root of the Python source tree  (…/clm-ml-jax/src)
_SRC_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(raw: str) -> str:
    """
    Convert a namelist file path to an absolute path.

    Namelist paths were written for the Fortran executable which lived
    inside CLM-ml_v1/offline_executable/ and used paths like
    ``../input_files/…``.  Under the Python layout the same files live
    under ``src/input_files/…``.

    Strategy (in order):
    1. If the path already exists as-is, return it.
    2. Strip leading ``../`` components and look under ``src/``.
    3. Return the original string unchanged (let the caller fail).
    """
    if not raw or raw.strip() in ("", " "):
        return raw
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return str(p)
    # Try as-is relative to cwd
    if p.exists():
        return str(p.resolve())
    # Strip leading '../' segments then resolve under _SRC_ROOT
    parts = p.parts
    # Drop leading '..' components
    while parts and parts[0] == "..":
        parts = parts[1:]
    candidate = _SRC_ROOT / Path(*parts)
    if candidate.exists():
        return str(candidate)
    return raw


def _path_list(value) -> list[str]:
    """A namelist value holding one path or a list of paths -> resolved paths (blanks
    dropped)."""
    values = [value] if isinstance(value, str) or value is None else list(value)
    return [_resolve_path(str(v).strip()) for v in values if v is not None and str(v).strip()]


def read_namelist(source) -> dict:
    """
    Parse a Fortran-style namelist from a file path or stdin.

    Mirrors Fortran:  read(5, nml=clm_inparm)
    in offline_driver/controlMod.F90.

    Args:
        source: File path string, or '-' / None to read from stdin.

    Returns:
        Dictionary of namelist group name -> parameter dict.
    """
    if source is None or source == "-":
        # Fortran:  ./prgm.exe < nl.CHATS7.05.2007
        text = sys.stdin.read()
        nml = f90nml.reads(text)
    else:
        # Fortran:  ./prgm.exe nl.CHATS7.05.2007  (alternative)
        nml = f90nml.read(source)
    return nml


def build_bounds(nml: dict) -> bounds_type:
    """
    Construct decomposition bounds from namelist parameters.

    The offline single-column case uses a single gridcell / column / patch,
    so all begin/end indices are 1.  Mirrors the stub decompMod in
    clm_src_main/decompMod.F90.

    Args:
        nml: Parsed namelist dictionary.

    Returns:
        A :class:`bounds_type` for the single-column offline run.
    """
    return bounds_type(
        begg=0,
        endg=0,
        begl=0,
        endl=0,
        begc=0,
        endc=0,
        begp=0,
        endp=0,
    )


@dataclasses.dataclass
class LegacyRun:
    """A run set up by :func:`setup_run`: everything the time loop needs."""

    params: dict  # namelist group
    bounds: bounds_type
    filt: object  # filterMod.clumpfilter
    insts: object  # clm_instMod.ClmInstances, initial conditions installed
    load_chunk: object  # (first_step, n_steps) -> ForcingChunk (reads the input files)
    ntimes: int
    tower_num: int
    iyear: int
    imonth: int
    dirout: str
    time_of: object  # 0-based run step -> StepTime
    acclimation: str = "running_mean"  # running_mean (decision 15) | run_mean (Fortran)
    acclimation_days: float = 10.0
    opts: MLOptions = DEFAULT_OPTIONS  # multilayer canopy options (static; decision 20)
    canopy_params: object = None  # the patch's CanopyParams, from the PFT tables at setup


@dataclasses.dataclass
class ForcingChunk:
    """Forcing for run steps ``first_step ... first_step + forcing.n_steps - 1``."""

    first_step: int
    forcing: object  # clm_ml.forcing.Forcing (with one halo record at each end)


@dataclasses.dataclass
class StepInputs:
    """Forcing and time of one step (0-based ``run_step``)."""

    now: object  # StepTime
    met: object  # tower meteorology record
    clm_record: object  # CLM history record
    forcing_bef: object  # AtmForcingRecord, previous step
    forcing_next: object  # AtmForcingRecord, next step


def setup_run(nml: dict, opts: Optional[MLOptions] = None) -> LegacyRun:
    """Configure the legacy modules from a namelist, initialize CLM, read the run's forcing
    and install initial conditions (CLMml_drv up to its time loop).

    ``opts``: the multilayer canopy options (``clm_ml``: from model.yaml). Without them (the
    ``clm-ml-offline`` command), the Fortran defaults with the namelist's ``met_type`` and
    ``dpai_min``, as Fortran's ``controlMod``."""
    # ------------------------------------------------------------------
    # 1. Namelist settings  (mirrors controlMod.F90 :: read(5, nml=...))
    # ------------------------------------------------------------------
    # Offline CLMml namelist group is typically &clmML_inparm
    params = nml.get("clmML_inparm", nml.get("clm_inparm", nml.get("clm_input", {})))

    tower_name = str(params.get("tower_name", "CHATS7"))
    start_ymd = int(params.get("start_ymd", 20070501))
    iyear = start_ymd // 10000
    imonth = (start_ymd // 100) % 100

    fin_tower = _resolve_path(str(params.get("fin_tower", "")))
    # Forcing split in time (run.yaml files with {year}/{month}): fin_tower_files lists the
    # tower files in order; without it, fin_tower is the only file.
    tower_files = _path_list(params.get("fin_tower_files")) or [fin_tower]
    fin_clm = _resolve_path(str(params.get("fin_clm", "")))
    fin_soil_adjust = _resolve_path(str(params.get("fin_soil_adjust", "")))
    fin_vegetation = _path_list(params.get("fin_vegetation"))
    fin_soil_water = _path_list(params.get("fin_soil_water"))
    dirout_raw = str(params.get("dirout", "")).strip()
    dirout = _resolve_path(dirout_raw) if dirout_raw else ""

    # fin2 = CLM history file   (used by clmData inside clm_driver)
    fin2 = fin_clm

    print(f"Site: {tower_name}  Year: {iyear}  Month: {imonth}", flush=True)

    # ------------------------------------------------------------------
    # 2. Set global run-control variables  (mirrors controlMod.F90)
    # ------------------------------------------------------------------
    controlMod.tower_site = tower_name
    controlMod.iyear = iyear
    controlMod.imonth = imonth

    # ------------------------------------------------------------------
    # 2b. Resolve tower index and propagate key settings to global modules
    # ------------------------------------------------------------------
    from clm_src_utils import clm_time_manager
    from clm_src_utils.clm_time_manager import step_time
    from multilayer_canopy import MLclm_varctl
    from offline_driver import TowerDataMod, clmSoilOptionMod

    tower_num = 0
    for i in range(1, int(TowerDataMod.ntower) + 1):
        if tower_name == str(TowerDataMod.tower_id[i]):
            tower_num = i
            break
    if tower_num == 0:
        raise ValueError(f"Tower site '{tower_name}' not found in TowerDataMod")

    TowerDataMod.tower_num = tower_num

    clmSoilOptionMod.clm_phys = str(params.get("clm_phys", clmSoilOptionMod.clm_phys))
    clmSoilOptionMod.nlev_soil_adjust = int(params.get("nlev_soil_adjust", 0))
    clmSoilOptionMod.fin_soil_adjust = fin_soil_adjust
    if opts is None:
        opts = DEFAULT_OPTIONS._replace(
            met_type=int(params.get("met_type", DEFAULT_OPTIONS.met_type)),
            dpai_min=float(params.get("dpai_min", DEFAULT_OPTIONS.dpai_min)),
        )
    MLclm_varctl.pftcon_val = int(params.get("pftcon_val", MLclm_varctl.pftcon_val))

    clm_time_manager.start_date_ymd = start_ymd
    clm_time_manager.start_date_tod = int(params.get("start_tod", 0))
    # The forcing data interval is the tower's; the model step (decision 13) defaults to it,
    # and the forcing is interpolated to it otherwise (clm_ml.forcing_interpolation).
    record_sec = int(TowerDataMod.tower_time[tower_num]) * 60
    clm_time_manager.dtstep = int(params.get("dtstep_seconds", record_sec))
    if record_sec % clm_time_manager.dtstep:
        raise ValueError(
            f"model step {clm_time_manager.dtstep} s must divide the forcing interval "
            f"{record_sec} s"
        )

    # ------------------------------------------------------------------
    # 3. Build decomposition bounds  (single column for offline case)
    # ------------------------------------------------------------------
    bounds = build_bounds(nml)

    # ------------------------------------------------------------------
    # 4. Compute ntimes
    # ------------------------------------------------------------------
    stop_option = str(params.get("stop_option", "ndays")).strip()
    stop_n = int(params.get("stop_n", 1))
    dtstep_sec = clm_time_manager.dtstep

    if stop_option == "ndays":
        ntimes = (stop_n * 86400) // max(dtstep_sec, 1)
    elif stop_option == "nsteps":
        ntimes = stop_n
    else:
        ntimes = stop_n

    # ------------------------------------------------------------------
    # 5. Initialize CLM  (mirrors CLMml_drv: InitializeRealize + setFilters)
    # ------------------------------------------------------------------
    # NOTE: clm_instMod must NOT be imported before InitializeRealize.
    # Importing clm_instMod triggers SoilStateType to capture nlevgrnd=-1
    # before ColumnType.py calls clm_varpar_init(). InitializeRealize calls
    # initialize1() which calls clm_varpar_init() first, so all imports of
    # clm_instMod and SoilStateType happen in the correct order inside the
    # lazy import chain of clm_initializeMod.
    from clm_src_main import filterMod as _filterMod

    InitializeRealize(bounds, opts.max_layers)
    # The patch's parameters, read from the PFT tables now: later steps use these, not the
    # tables (which another model in this process may replace).
    from clm_src_main.PatchType import patch
    from multilayer_canopy.MLCanopyParamsMod import canopy_params_from_tables

    canopy_params = canopy_params_from_tables(int(np.asarray(patch.itype)[bounds.begp]))

    # clm_instMod is now safe to import (nlevgrnd correctly set)
    from clm_src_main import clm_instMod

    # Filters and model state are created once here and then carried explicitly
    # through the time loop (ModelAdvance takes and returns them).
    filt = _filterMod.setFilters(_filterMod.filter)
    _filterMod.filter = filt
    st = clm_instMod.get_instances()

    print("Initialization complete.", flush=True)

    # ------------------------------------------------------------------
    # 6. Pre-loop setup  (mirrors CLMml_drv lines 84-120)
    # ------------------------------------------------------------------
    from clm_ml.legacy.forcing import (
        tower_met_at,
    )
    from offline_driver.CLMml_driver import (
        SoilInit,
        TowerVeg,
        init_acclim,
    )
    from offline_driver.TowerMetMod import TowerFiles

    # 6-pre. Time axis of the run, and the matching time slice of the CLM history file
    #     for every step. Fortran CLMml_drv computed the history file's first calendar day
    #     by temporarily overwriting the run start date ("hack", lines 97-116); here it is
    #     computed directly from the history file's start date.
    run_start_tod = int(params.get("start_tod", 0))
    clm_start_ymd = int(params.get("clm_start_ymd", start_ymd))
    clm_start_tod = int(params.get("clm_start_tod", 0))

    def _time(run_step: int):
        """StepTime of a 0-based run step."""
        return step_time(start_ymd, run_start_tod, dtstep_sec, run_step + 1)

    # The CLM history file has its own interval (from its time axis): a model step uses the
    # time slice whose interval contains the step's end (slice s ends at the history start
    # + s intervals). With the history interval equal to the model step, slice k + 1 for step k.
    import netCDF4 as _nc

    with _nc.Dataset(fin2) as _ds:
        _t = _ds.variables["time"]
        history_sec = int(round(float(_t[1] - _t[0]) * 86400.0))
    clm_start = datetime.datetime.strptime(str(clm_start_ymd), "%Y%m%d") + datetime.timedelta(
        seconds=clm_start_tod
    )
    _run_start = datetime.datetime.strptime(str(start_ymd), "%Y%m%d") + datetime.timedelta(
        seconds=run_start_tod
    )

    def _history_index(run_step: int) -> int:
        """1-based CLM history time slice of a 0-based run step."""
        step_end = _run_start + datetime.timedelta(seconds=(run_step + 1) * dtstep_sec)
        elapsed = (step_end - clm_start).total_seconds()
        return int(-(-elapsed // history_sec))  # ceil: the slice whose interval contains it

    soil_init_time_indx = _history_index(0)

    # 6-load. Forcing is read one chunk at a time (with one halo record at each end), so
    #     memory does not depend on run length (plan §6.9).
    run_start = datetime.datetime.strptime(str(start_ymd), "%Y%m%d") + datetime.timedelta(
        seconds=run_start_tod
    )
    tower_series = TowerFiles.open(tower_files, record_sec)
    first_record = tower_series.first_record(run_start)
    tower_height = float(TowerDataMod.tower_ht[tower_num])

    from clm_ml.legacy.forcing import TowerForcingSource

    source = TowerForcingSource(
        tower_file=tower_series,
        clm_history_file=fin2,
        soil_adjust_file=fin_soil_adjust,
        first_record=first_record,
        n_steps=ntimes,
        history_index=_history_index,
        tower_height=tower_height,
        record_seconds=record_sec,
        step_seconds=dtstep_sec,
        run_start=run_start,
        lat=float(TowerDataMod.tower_lat[tower_num]),
        lon=float(TowerDataMod.tower_lon[tower_num]),
        vegetation_file=tuple(fin_vegetation) or None,
        soil_water_file=tuple(fin_soil_water) or None,
        shortwave=str(params.get("met_shortwave", "coszen")).strip(),
        precipitation=str(params.get("met_precipitation", "constant")).strip(),
    )

    def load_chunk(first_step: int, n_steps: int) -> ForcingChunk:
        return ForcingChunk(first_step=first_step, forcing=source.chunk(first_step, n_steps))

    def run_tower_met(chunk_steps: int = 1000):
        """Each step's tower meteorology, read chunk by chunk (index t + 1 skips the
        chunk's leading halo record)."""
        for first in range(0, ntimes, chunk_steps):
            chunk = load_chunk(first, min(chunk_steps, ntimes - first))
            for t in range(chunk.forcing.n_steps):
                yield tower_met_at(chunk.forcing, t + 1)

    # 6a. Acclimation temperature. running_mean (default, decision 15): starts at the first
    #     step's air temperature and is updated every step (advance_acclimation). run_mean
    #     (Fortran CLM-ml v2, validation only): the mean over the whole run, summed over the
    #     run's forcing read chunk by chunk. (init_acclim also sets the reference pressure
    #     from the first record.)
    acclimation = str(params.get("acclimation", "running_mean")).strip()
    acclimation_days = float(params.get("acclimation_days", 10.0))
    if acclimation not in ("running_mean", "run_mean"):
        raise ValueError(f"acclimation must be running_mean or run_mean, got {acclimation!r}")
    from offline_driver.TowerMetMod import set_tower_location

    set_tower_location(tower_num, bounds.begp, bounds.endp)
    atm2lnd, wateratm2lndbulk, temperature, frictionvel, mlcanopy = init_acclim(
        run_tower_met() if acclimation == "run_mean" else iter([next(run_tower_met(1))]),
        tower_num,
        ntimes if acclimation == "run_mean" else 1,
        bounds.begp,
        bounds.endp,
        st.atm2lnd,
        st.wateratm2lndbulk,
        st.temperature,
        st.frictionvel,
        st.mlcanopy,
    )
    st = st._replace(
        atm2lnd=atm2lnd,
        wateratm2lndbulk=wateratm2lndbulk,
        temperature=temperature,
        frictionvel=frictionvel,
        mlcanopy=mlcanopy,
    )

    # 6aa. Orbital parameters for this year — mirrors CLMml_drv lines 80-81
    from clm_share.shr_orb_mod import shr_orb_params as _shr_orb_params
    from clm_src_utils import clm_varorb as _clm_varorb

    _eccen, _obliq, _mvelp, _obliqr, _lambm0, _mvelpp = _shr_orb_params(iyear)
    _clm_varorb.eccen = _eccen
    _clm_varorb.obliqr = _obliqr
    _clm_varorb.lambm0 = _lambm0
    _clm_varorb.mvelpp = _mvelpp

    # 6b. Tower vegetation properties
    canopystate, mlcanopy = TowerVeg(
        tower_num,
        bounds.begp,
        bounds.endp,
        st.canopystate,
        st.mlcanopy,
    )
    st = st._replace(canopystate=canopystate, mlcanopy=mlcanopy)

    # 6d. Soil temperature and moisture initialisation
    waterstatebulk, temperature = SoilInit(
        fin2,
        soil_init_time_indx,
        bounds.begc,
        bounds.endc,
        st.soilstate,
        st.waterstatebulk,
        st.temperature,
    )
    st = st._replace(waterstatebulk=waterstatebulk, temperature=temperature)

    # 6e. Per-site data the time step reads (Phase 6 step 2): from the subgrid modules, now
    #     final, into the containers (so into the template, an input of the compiled chunk)
    from clm_src_main.SiteType import site_from_modules

    st = st._replace(site=site_from_modules())

    return LegacyRun(
        params=params,
        bounds=bounds,
        filt=filt,
        insts=st,
        load_chunk=load_chunk,
        acclimation=acclimation,
        acclimation_days=acclimation_days,
        opts=opts,
        canopy_params=canopy_params,
        ntimes=ntimes,
        tower_num=tower_num,
        iyear=iyear,
        imonth=imonth,
        dirout=dirout,
        time_of=_time,
    )


def step_inputs(run: LegacyRun, chunk: ForcingChunk, run_step: int) -> StepInputs:
    """Time and forcing of a 0-based run step (inside ``chunk``), including the previous
    and next forcing records for the multilayer canopy's 3-point time interpolation
    (Fortran *_bef/_next_forcing; TowerMetNext)."""
    from clm_ml.legacy.forcing import (
        atm_record_current,
        atm_record_next,
        clm_record_at,
        tower_met_at,
    )

    forcing = chunk.forcing
    t_prev, t_curr, t_next = forcing.window_indices(run_step - chunk.first_step)
    n_patch = run.bounds.endp + 1
    return StepInputs(
        now=run.time_of(run_step),
        met=tower_met_at(forcing, t_curr),
        clm_record=clm_record_at(forcing, t_curr),
        forcing_bef=atm_record_current(forcing, t_prev, n_patch),
        forcing_next=atm_record_next(forcing, t_next, n_patch),
    )


def advance(
    run: LegacyRun, insts, filt, inputs: StepInputs, initialize: bool, params=None, grid=None
):
    """One time step: install the step's tower meteorology, then ModelAdvance.
    ``grid``: the patch's layer indices (``patch_setup``); required when ``insts`` are traced
    (a compiled chunk), found from ``insts`` otherwise. Returns ``(insts, filt)``."""
    from offline_driver.TowerMetMod import TowerMetCurr

    atm2lnd, wateratm2lndbulk, frictionvel = TowerMetCurr(
        inputs.met,
        run.tower_num,
        run.bounds.begp,
        run.bounds.endp,
        insts.atm2lnd,
        insts.wateratm2lndbulk,
        insts.frictionvel,
    )
    insts = insts._replace(
        atm2lnd=atm2lnd, wateratm2lndbulk=wateratm2lndbulk, frictionvel=frictionvel
    )
    insts, filt = ModelAdvance(
        run.bounds,
        insts,
        filt,
        inputs.clm_record,
        inputs.now,
        inputs.forcing_bef,
        inputs.forcing_next,
        initialize=initialize,
        params=run.canopy_params if params is None else params,
        # the first step sets up the layers (all options); later steps get only the options
        # that select physics code (setup-only fields at defaults: one program for patches
        # with different layer settings)
        opts=run.opts if initialize else run.opts.step_options(),
        grid=grid,
    )
    # Fortran's conservation checks (recorded in the step, plan §6.8): checked here when the
    # step runs on concrete values; after each chunk when steps are compiled.
    errors = insts.mlcanopy.balance_error
    if is_concrete(errors):
        check_balance(errors, f"time step {inputs.now.nstep}")
    return insts, filt


def month_chunks(run: LegacyRun, first_step: int) -> list:
    """``(first_step, n_steps)`` chunks covering run steps ``first_step ... ntimes - 1``, one
    per calendar month of the steps' start times (plan decision 17)."""
    chunks = []
    for k in range(first_step, run.ntimes):
        t = run.time_of(k)
        start = datetime.datetime(t.year, t.month, t.day) + datetime.timedelta(
            seconds=t.tod - t.dtime
        )
        key = (start.year, start.month)
        if chunks and chunks[-1][0] == key:
            chunks[-1][2] += 1
        else:
            chunks.append([key, k, 1])
    return [(k, n) for _, k, n in chunks]


def advance_acclimation(run: LegacyRun, state, inputs: StepInputs):
    """The state with the acclimation temperature's running mean advanced by this step's
    forcing air temperature (decision 15): ``mean += (T - mean) / min(n, N)``, with ``n``
    the steps averaged so far (this one included) and ``N`` the steps in
    ``acclimation_days``: the mean of all steps so far for the first ``N`` steps, then a
    running mean with an ``N``-step time scale (CLM's ``runmean`` accumulation). Unchanged
    for ``acclimation = run_mean``. Applied before every step after the first."""
    if run.acclimation != "running_mean":
        return state
    c = state.canopy
    period = run.acclimation_days * 86400.0 / run.time_of(0).dtime
    n = c.acclimation_steps + 1.0
    t = jnp.asarray(inputs.met.tref, dtype=jnp.float64)  # tower air temperature (K)
    mean = c.acclimation_temperature + (t - c.acclimation_temperature) / jnp.minimum(n, period)
    return state._replace(canopy=c._replace(acclimation_temperature=mean, acclimation_steps=n))


def stack_step_inputs(inputs: list) -> tuple:
    """Per-step inputs stacked along a leading time axis (the ``xs`` of a chunk's scan)."""
    return jax.tree.map(
        lambda *x: jnp.stack([jnp.asarray(v) for v in x]),
        *[(i.now, i.met, i.clm_record, i.forcing_bef, i.forcing_next) for i in inputs],
    )


def make_chunk_runner(run: LegacyRun, template, filt, outputs: str = "containers"):
    """A compiled function ``(state, xs) -> (state, per-step outputs)`` that advances a
    chunk of steps as one ``jax.lax.scan`` (plan Phase 4). ``xs`` is from
    :func:`stack_step_inputs`; the per-step outputs are ``(state, mlcanopy,
    temperature)`` after each step (the containers are for the output writer), or only the
    state with ``outputs="state"`` (e.g. for ensembles). Steps are rebuilt from ``template``
    plus the carried state, as in the step-by-step loop.

    The compiled function takes an optional third argument, the patch's ``CanopyParams``
    (default: from the PFT tables), so ``jax.vmap`` over it runs a parameter ensemble and
    ``jax.grad`` differentiates with respect to it."""
    dtime = run.time_of(0).dtime  # static: the model step
    grid = _grid_of(template, filt)

    def body(state, x, params):
        now, met, clm_record, forcing_bef, forcing_next = x
        inputs = StepInputs(
            now=now._replace(dtime=dtime),
            met=met,
            clm_record=clm_record,
            forcing_bef=forcing_bef,
            forcing_next=forcing_next,
        )
        state = advance_acclimation(run, state, inputs)
        insts, _ = advance(
            run,
            instances_with_state(template, state),
            filt,
            inputs,
            initialize=False,
            params=params,
            grid=grid,
        )
        state = state_from_instances(insts, state)
        if outputs == "state":
            return state, state
        return state, (state, insts.mlcanopy, insts.temperature)

    default_params = run.canopy_params

    def run_chunk(state, xs, params=default_params):
        return jax.lax.scan(lambda s, x: body(s, x, params), state, xs)

    return jax.jit(run_chunk)


#: jax.checkpoint on the step of compiled chunks (make_stream_runner), as in jax-gcm; a
#: switch for experiments. (Gradients over a chunk need XLA's older CPU runtime with JAX
#: 0.6.2: see clm_ml.model.)
CHECKPOINT_STEP = True


def _patch_step(run: LegacyRun, filt, grid, names, max_layers: int):
    """One patch's time step for compiled chunks: ``step(template, state, x, params) ->
    (state, {variable: value}, errors)``, with ``x`` one step's ``(now, met, clm_record,
    forcing_bef, forcing_next)``, the output variables ``names`` (``clm_ml.outputs``,
    computed with jax.numpy; leading axis the legacy patch axis of length 1) and ``errors``
    the step's largest error of each Fortran conservation check. Used by the one-patch
    (:func:`make_stream_runner`) and batched (:func:`make_batch_runner`) chunks."""
    from clm_ml.outputs import VARIABLES_BY_NAME, _Raw
    from clm_src_main.clm_varpar import nlevgrnd

    dtime = run.time_of(0).dtime

    def variables(insts) -> dict:
        m = insts.mlcanopy
        raw = {f"mlcanopy.{f}": getattr(m, f)[None] for f in m._fields}
        r = _Raw(raw, max_layers, xp=jnp)
        out = {}
        for n in names:
            if n == "soil_temperature":  # (patch == column) soil layers 1..nlevgrnd
                out[n] = insts.temperature.t_soisno_col[:, 1 : nlevgrnd + 1]
            elif "." in n:  # raw legacy field, e.g. "mlcanopy.tair_profile" (validation)
                container, field = n.split(".", 1)
                out[n] = jnp.asarray(getattr(getattr(insts, container), field), dtype=float)
            else:
                out[n] = VARIABLES_BY_NAME[n].compute(r)[0]
        return out

    def step(template, state, x, params):
        now, met, clm_record, forcing_bef, forcing_next = x
        inputs = StepInputs(
            now=now._replace(dtime=dtime),
            met=met,
            clm_record=clm_record,
            forcing_bef=forcing_bef,
            forcing_next=forcing_next,
        )
        state = advance_acclimation(run, state, inputs)
        insts, _ = advance(
            run,
            instances_with_state(template, state),
            filt,
            inputs,
            initialize=False,
            params=params,
            grid=grid,
        )
        errors = insts.mlcanopy.balance_error.max(axis=0)
        return state_from_instances(insts, state), variables(insts), errors

    return step


def make_stream_runner(run: LegacyRun, template, filt, streams, max_layers: int):
    """A compiled function ``(state, xs, slots, buffers[, params]) -> (state, buffers,
    errors)``
    advancing a chunk as one ``jax.lax.scan`` and accumulating output streams inside it
    (plan §6.9): each step's output variables (``clm_ml.outputs``, computed with jax.numpy)
    are added into the slot of the stream record the step belongs to. ``streams`` are
    ``clm_ml.output_streams.StreamSpec``; ``slots[i]`` is the per-step record index of
    stream ``i`` (``chunk_records``), ``buffers[i]`` its ``init_buffers``. No per-step
    values leave the device. ``errors``: the chunk's largest error of each Fortran
    conservation check (``MLCanopyFluxesType.BALANCE_CHECKS``), for ``check_balance``.
    One patch; :func:`make_batch_runner` runs many."""
    from clm_ml import output_streams as S

    names = sorted({n for spec in streams for n in spec.variables})
    default_params = run.canopy_params
    patch_step = _patch_step(run, filt, _grid_of(template, filt), names, max_layers)

    def step(template, carry, x):
        state, buffers, errors = carry
        xs, slots, params = x
        state, values, step_errors = patch_step(template, state, xs, params)
        buffers = tuple(
            S.accumulate(spec.statistic, b, slot, {n: values[n] for n in spec.variables})
            for spec, b, slot in zip(streams, buffers, slots)
        )
        errors = jnp.maximum(errors, step_errors)
        return (state, buffers, errors), None

    # As in jax-gcm: jax.checkpoint on the step (if CHECKPOINT_STEP), so a gradient over a
    # chunk stores only the carry (state and stream buffers) at each step and recomputes the
    # step's intermediate values in the backward pass. No effect on forward results.
    checkpointed_step = jax.checkpoint(step) if CHECKPOINT_STEP else step

    # The patch's structure and the containers' other fields (``template``) are an input of
    # the compiled chunk, not constants baked into it (plan Phase 5b): another patch with the
    # same layer indices reuses the program.
    def run_chunk(state, xs, slots, buffers, params=default_params, template=template):
        n = jax.tree.leaves(xs)[0].shape[0]
        per_step_params = jax.tree.map(lambda p: jnp.broadcast_to(p, (n,) + p.shape), params)
        errors = jnp.zeros(len(BALANCE_CHECKS))
        (state, buffers, errors), _ = jax.lax.scan(
            lambda c, x: checkpointed_step(template, c, x),
            (state, buffers, errors),
            (xs, slots, per_step_params),
        )
        return state, buffers, errors

    return jax.jit(run_chunk)


@dataclasses.dataclass
class PatchBatch:
    """Patches run together (plan Phase 6 step 3): each set up as a one-patch legacy run
    (``runs``), then advanced with one compiled step batched over the patch axis, C3 and C4
    patches in two groups (decision 22, step 4). All share one clock (decision 24) and the
    options; ``runs[0]`` provides them.
    Patches are in domain order; ``gridcell_index``/``weight`` map them to gridcells."""

    runs: list  # LegacyRun per patch
    gridcell_index: np.ndarray  # (P,) int
    weight: np.ndarray  # (P,) area fraction of the gridcell
    n_gridcells: int
    names: list  # per patch, for messages
    grids: list  # GridInfo per patch (patch index, C3/C4), from patch_grid right after setup
    # Each patch's first (initializing) step, ``(insts, filt)``, run right after its setup
    # (it reads the patch's PFT and tables from module state, which the next setup replaces)
    firsts: Optional[list] = None
    # The domain's patches and gridcells come first; any after them are padding (copies with
    # weight 0 on padding gridcells, :meth:`padded`), dropped from the outputs
    n_real: Optional[int] = None
    n_real_gridcells: Optional[int] = None

    def __post_init__(self):
        if self.n_real is None:
            self.n_real = len(self.runs)
        if self.n_real_gridcells is None:
            self.n_real_gridcells = self.n_gridcells

    @classmethod
    def single(cls, run: LegacyRun) -> "PatchBatch":
        return cls([run], np.zeros(1, dtype=int), np.ones(1), 1, ["patch 0"], [patch_grid(run)])

    @staticmethod
    def first_step(run: LegacyRun, grid, params=None):
        """A patch's first (initializing) step: ``(insts, filt)``."""
        inputs = step_inputs(run, run.load_chunk(0, 1), 0)
        return advance(run, run.insts, run.filt, inputs, True, params, grid)

    @property
    def run(self) -> LegacyRun:
        return self.runs[0]

    @property
    def n_patches(self) -> int:
        return len(self.runs)

    @property
    def groups(self) -> list:
        """``[(GridInfo, patch indices), ...]``: the C3 group, then the C4 group (decision 22),
        each present only if it has patches. The patch index ``GridInfo.p`` is the same for
        every patch (each is a one-patch legacy run)."""
        out = []
        for c3 in (True, False):
            idx = np.array([p for p, g in enumerate(self.grids) if g.c3 == c3], dtype=int)
            if idx.size:
                out.append((self.grids[int(idx[0])], idx))
        return out

    @property
    def gridcell_patch(self) -> list:
        """The first patch of each gridcell (whose run reads the gridcell's forcing); a padding
        gridcell without patches reads the last real gridcell's forcing."""
        out = []
        for g in range(self.n_gridcells):
            on_g = np.flatnonzero(self.gridcell_index == g)
            out.append(int(on_g[0]) if on_g.size else out[self.n_real_gridcells - 1])
        return out

    def padded(self, patch_count: Optional[int], gridcell_count: Optional[int]) -> "PatchBatch":
        """This batch padded to ``patch_count`` patches and ``gridcell_count`` gridcells, so
        that domains of different sizes share one compiled program (Phase 6 step 3): padding
        gridcells read the last real gridcell's forcing; padding patches are copies of the
        last real patch with weight 0, on the last padding gridcell (or the last real one)."""
        n_p, n_g = patch_count or self.n_patches, gridcell_count or self.n_gridcells
        if n_p < self.n_patches or n_g < self.n_gridcells:
            raise ValueError(
                f"execution.patch_count/gridcell_count ({n_p}, {n_g}) is smaller than the "
                f"domain ({self.n_patches} patches, {self.n_gridcells} gridcells)"
            )
        k = n_p - self.n_patches
        return dataclasses.replace(
            self,
            runs=self.runs + [self.runs[-1]] * k,
            gridcell_index=np.concatenate([self.gridcell_index, np.full(k, n_g - 1)]),
            weight=np.concatenate([self.weight, np.zeros(k)]),
            n_gridcells=n_g,
            names=self.names + [f"padding {i}" for i in range(k)],
            grids=self.grids + [self.grids[-1]] * k,
            firsts=None if self.firsts is None else self.firsts + [self.firsts[-1]] * k,
            n_real=self.n_patches,
            n_real_gridcells=self.n_gridcells,
        )

    def params(self):
        """The patches' parameters (``CanopyParams``) stacked along the patch axis."""
        return stack_patches([r.canopy_params for r in self.runs])


def patch_grid(run: LegacyRun):
    """The patch's ``GridInfo`` (patch index and static C3/C4 flag). Call it right after the
    patch's setup: it reads the patch's PFT from the CLM subgrid module, which the next
    patch's setup replaces."""
    from multilayer_canopy.MLCanopyParamsMod import patch_setup

    grid, _ = patch_setup(run.bounds.begp, run.insts.mlcanopy)
    return grid


def stack_patches(trees: list):
    """Pytrees of single patches stacked along a new leading patch axis."""
    return jax.tree.map(lambda *x: jnp.stack([jnp.asarray(v) for v in x]), *trees)


def unbatch_patch_axis(x, n_patches: int):
    """A batched output value ``(P, 1, ...)`` (the legacy patch axis of one patch) as
    ``(P, ...)``."""
    if x.shape[1] != 1:
        raise ValueError(f"expected a legacy patch axis of length 1, got shape {x.shape}")
    return x.reshape((n_patches,) + x.shape[2:])


def make_batch_runner(batch: PatchBatch, template, filt, streams, max_layers: int, gridcell_vars):
    """A compiled function ``(state, xg, xp, slots, buffers, params, weight, alive, gidx) ->
    (state, buffers, errors)`` advancing all patches of ``batch`` over a chunk as one ``jax.lax.scan`` whose
    step is the one-patch step (:func:`_patch_step`) ``vmap``ped over the patch axis (plan
    Phase 6 step 3).

    ``template``, ``state`` and ``params`` have a leading patch axis. The forcing comes per
    gridcell (``xg``: ``(now, met, forcing_bef, forcing_next)``, axes (step, gridcell, ...))
    and is gathered to the patches; the CLM history record per patch (``xp``: (step, patch,
    ...)). Stream buffers hold every variable per patch ``(record, patch, ...)`` and, for
    ``gridcell_vars[i]`` of stream ``i``, the area-weighted sum over each gridcell's patches
    formed at every step (``gridcell:<variable>``, ``(record, gridcell, ...)``), so every
    statistic (also min and max) is right at gridcell level; ``weight`` (patch area
    fractions) is an input, so the gridcell sums are differentiable with respect to it; the
    patch-gridcell map ``gidx`` is an input too, so domains with the same numbers of patches
    and gridcells share the program.
    ``errors``: ``(patch, check)``.

    ``alive`` ``(patch,)`` marks the patches still in the run (decision 25, ``exclude``). An
    excluded patch, or one whose value is not finite at a step, is left out of its gridcell's
    sum and the other patches' weights are renormalized; its parameters enter through
    ``where(alive, p, stop_gradient(p))``, so a failed patch never reaches a gridcell value or
    the gradient of a shared PFT parameter. With every patch fine, the plain weighted sum is
    used (same values as without the mask)."""
    from clm_ml import output_streams as S

    names = sorted({n for spec in streams for n in spec.variables})
    # C3/C4 groups (decision 22, step 4): each runs its own pathway's code, vmapped over its
    # patches; one group (the usual case) runs over all patches without gathering
    groups = batch.groups
    group_steps = [_patch_step(batch.run, filt, grid, names, max_layers) for grid, _ in groups]
    n_p, n_g = batch.n_patches, batch.n_gridcells

    def batched_step(templates, state, x_patch, params):
        """The step of every patch: per group, vmapped; results back in patch order."""
        if len(groups) == 1:
            return jax.vmap(group_steps[0])(templates[0], state, x_patch, params)
        out_state, out_values, out_errors = state, None, None
        for (_, idx), step_g, tmpl in zip(groups, group_steps, templates):
            take = lambda t: jax.tree.map(lambda a: jnp.take(a, idx, 0), t)  # noqa: E731
            st, vals, err = jax.vmap(step_g)(tmpl, take(state), take(x_patch), take(params))
            if out_values is None:  # patch-order arrays of the right shapes
                out_values = {k: jnp.zeros((n_p,) + v.shape[1:], v.dtype) for k, v in vals.items()}
                out_errors = jnp.zeros((n_p,) + err.shape[1:], err.dtype)
            out_state = jax.tree.map(lambda a, b: a.at[idx].set(b), out_state, st)
            out_values = {k: out_values[k].at[idx].set(v) for k, v in vals.items()}
            out_errors = out_errors.at[idx].set(err)
        return out_state, out_values, out_errors

    def gridcell_sum(v, weight, alive, gidx):  # (P, ...) -> (G, ...): area-weighted sum
        def per(a):  # (P,) -> broadcast over v's trailing axes
            return a.reshape((-1,) + (1,) * (v.ndim - 1))

        def seg(a):
            return jax.ops.segment_sum(a, gidx, num_segments=n_g)

        ok = alive & jnp.all(jnp.isfinite(v.reshape(n_p, -1)), axis=1)
        v_safe = jnp.where(per(ok), v, 0.0)  # NaN never enters a sum (nor its gradient)
        plain = seg(v_safe * per(weight))
        w_ok = jnp.where(ok, weight, 0.0)
        den = seg(w_ok)
        den_b = den.reshape((-1,) + (1,) * (v.ndim - 1))
        renorm = seg(v_safe * per(w_ok)) / jnp.where(den_b > 0.0, den_b, 1.0)
        renorm = jnp.where(den_b > 0.0, renorm, jnp.nan)  # every patch of the gridcell out
        some_out = (seg((~ok).astype(jnp.int32)) > 0).reshape(den_b.shape)
        return jnp.where(some_out, renorm, plain)

    def step(template, carry, x):
        state, buffers, errors = carry
        xg, xp, slots, params, weight, alive, gidx = x
        now, met, forcing_bef, forcing_next = jax.tree.map(lambda a: jnp.take(a, gidx, 0), xg)
        x_patch = (now, met, xp, forcing_bef, forcing_next)
        state, values, step_errors = batched_step(template, state, x_patch, params)
        values = {n: unbatch_patch_axis(v, n_p) for n, v in values.items()}
        new_buffers = []
        for spec, b, slot, gvars in zip(streams, buffers, slots, gridcell_vars):
            vals = {n: values[n] for n in spec.variables}
            vals.update(
                {f"gridcell:{n}": gridcell_sum(values[n], weight, alive, gidx) for n in gvars}
            )
            new_buffers.append(S.accumulate(spec.statistic, b, slot, vals))
        errors = jnp.maximum(errors, step_errors)
        return (state, tuple(new_buffers), errors), None

    checkpointed_step = jax.checkpoint(step) if CHECKPOINT_STEP else step

    def run_chunk(state, xg, xp, slots, buffers, params, weight, alive, gidx, template=template):
        n = jax.tree.leaves(xg)[0].shape[0]
        params = jax.tree.map(
            lambda p: jnp.where(
                alive.reshape((-1,) + (1,) * (p.ndim - 1)), p, jax.lax.stop_gradient(p)
            ),
            params,
        )
        per_step = jax.tree.map(
            lambda p: jnp.broadcast_to(p, (n,) + p.shape), (params, weight, alive, gidx)
        )
        errors = jnp.zeros((n_p, len(BALANCE_CHECKS)))
        # the groups' templates, gathered once per chunk (not at every step)
        templates = (
            (template,)
            if len(groups) == 1
            else tuple(jax.tree.map(lambda a: jnp.take(a, idx, 0), template) for _, idx in groups)
        )
        (state, buffers, errors), _ = jax.lax.scan(
            lambda c, x: checkpointed_step(templates, c, x),
            (state, buffers, errors),
            (xg, xp, slots) + per_step,
        )
        return state, buffers, errors

    return jax.jit(run_chunk)


def read_batch_inputs(batch: PatchBatch, first_step: int, n_steps: int):
    """A chunk's forcing for :func:`make_batch_runner`: ``xg`` (time, meteorology and the
    canopy forcing records, read once per gridcell, axes (step, gridcell, ...)) and ``xp``
    (the CLM history record per patch, axes (step, patch, ...))."""
    steps = range(first_step, first_step + n_steps)
    chunks = {}

    def inputs_of(p):
        run = batch.runs[p]
        if id(run) not in chunks:  # padding patches share their run: read once
            chunk = run.load_chunk(first_step, n_steps)
            chunks[id(run)] = [step_inputs(run, chunk, k) for k in steps]
        return chunks[id(run)]

    per_gridcell = []
    for p in batch.gridcell_patch:
        ins = inputs_of(p)
        per_gridcell.append(
            stack_step_inputs_parts(ins, ("now", "met", "forcing_bef", "forcing_next"))
        )
    per_patch = [
        stack_step_inputs_parts(inputs_of(p), ("clm_record",))[0] for p in range(batch.n_patches)
    ]
    xg = jax.tree.map(lambda *x: jnp.stack(x, axis=1), *per_gridcell)
    xp = jax.tree.map(lambda *x: jnp.stack(x, axis=1), *per_patch)
    return xg, xp


def stack_step_inputs_parts(inputs: list, parts: tuple) -> tuple:
    """The ``parts`` (StepInputs field names) of per-step inputs, stacked along a leading
    time axis."""
    return jax.tree.map(
        lambda *x: jnp.stack([jnp.asarray(v) for v in x]),
        *[tuple(getattr(i, f) for f in parts) for i in inputs],
    )


def _grid_of(template, filt):
    """The patch's layer indices and C3/C4 flag (``MLCanopyParamsMod.patch_setup``), found
    once from the concrete template, outside the compiled chunk."""
    from multilayer_canopy.MLCanopyParamsMod import patch_setup

    grid, _ = patch_setup(int(np.asarray(filt.exposedvegp)[0]), template.mlcanopy)
    return grid


def output_shapes(run: LegacyRun, template, names, max_layers: int) -> dict:
    """Per-step shape of each output variable (for ``init_buffers``)."""
    from clm_ml.outputs import VARIABLES_BY_NAME, _Raw
    from clm_src_main.clm_varpar import nlevgrnd

    m = template.mlcanopy
    r = _Raw({f"mlcanopy.{f}": np.asarray(getattr(m, f))[None] for f in m._fields}, max_layers)
    shapes = {}
    for n in names:
        if n == "soil_temperature":
            shapes[n] = np.asarray(template.temperature.t_soisno_col)[:, 1 : nlevgrnd + 1].shape
        elif "." in n:
            container, field = n.split(".", 1)
            shapes[n] = np.shape(getattr(getattr(template, container), field))
        else:
            shapes[n] = np.shape(VARIABLES_BY_NAME[n].compute(r))[1:]
    return shapes


def step_values(insts, names, max_layers: int) -> dict:
    """Output variables of one step from its containers (numpy), as the stream runner
    computes them inside a chunk; used for the first (initializing) step."""
    from clm_ml.outputs import VARIABLES_BY_NAME, _Raw
    from clm_src_main.clm_varpar import nlevgrnd

    m = insts.mlcanopy
    r = _Raw({f"mlcanopy.{f}": np.asarray(getattr(m, f))[None] for f in m._fields}, max_layers)
    out = {}
    for n in names:
        if n == "soil_temperature":
            out[n] = np.asarray(insts.temperature.t_soisno_col)[:, 1 : nlevgrnd + 1]
        elif "." in n:
            container, field = n.split(".", 1)
            out[n] = np.asarray(getattr(getattr(insts, container), field), dtype=float)
        else:
            out[n] = np.asarray(VARIABLES_BY_NAME[n].compute(r), dtype=float)[0]
    return out


def step_starts(run: LegacyRun, first_step: int, n_steps: int) -> list:
    """Start times (datetime) of run steps ``first_step ... first_step + n_steps - 1``."""
    out = []
    for k in range(first_step, first_step + n_steps):
        t = run.time_of(k)
        out.append(
            datetime.datetime(t.year, t.month, t.day) + datetime.timedelta(seconds=t.tod - t.dtime)
        )
    return out


#: State fields that are temperatures (K), checked against a plausible range
_TEMPERATURE_FIELDS = (
    "air_temperature", "leaf_temperature", "ground_temperature", "acclimation_temperature",
    "temperature",
)  # fmt: skip
_TEMPERATURE_RANGE = (150.0, 350.0)


def check_health(state, template_state, where: str) -> None:
    """Stop the run if the state has non-finite values, or temperatures outside
    150-350 K, at any position that is meaningful in the template state (finite and not a
    fill value; for temperatures also nonzero, since 0 marks unused entries). ``where`` (e.g. the chunk's dates) goes into the error message.
    """
    problems = health_problems(state, template_state)
    if problems:
        raise RuntimeError(f"unhealthy model state after {where}: " + "; ".join(problems))


def health_problems(state, template_state) -> list:
    """The problems :func:`check_health` reports (empty: healthy)."""
    problems = []
    for part in ("canopy", "soil"):
        for name, value in getattr(state, part)._asdict().items():
            ref = np.asarray(getattr(getattr(template_state, part), name), dtype=float)
            x = np.asarray(value, dtype=float)
            meaningful = np.isfinite(ref) & (np.abs(ref) < 1e30)
            bad = meaningful & ~np.isfinite(x)
            if bad.any():
                problems.append(f"{part}.{name}: {int(bad.sum())} non-finite values")
            elif name in _TEMPERATURE_FIELDS and (meaningful & (ref > 0)).any():
                used = meaningful & (ref > 0)  # 0 marks unused entries (e.g. leafless layers)
                lo, hi = float(x[used].min()), float(x[used].max())
                if lo < _TEMPERATURE_RANGE[0] or hi > _TEMPERATURE_RANGE[1]:
                    problems.append(f"{part}.{name}: range {lo:.1f}-{hi:.1f} K")
    return problems


def restart_boundary(end: datetime.datetime, every) -> bool:
    """Whether a chunk ending at ``end`` ends on a ``restart_every`` boundary
    (``(n, "month")`` or ``(n, "year")``; None: never)."""
    if every is None or (end.day, end.hour, end.minute, end.second) != (1, 0, 0, 0):
        return False
    n, unit = every
    if unit == "month":
        return (end.year * 12 + end.month - 1) % n == 0
    return end.month == 1 and end.year % n == 0


def write_restart(path: Path, content: dict) -> None:
    """Write ``content`` beside the old restart and rename it over it, keeping the previous
    one as ``<path>.prev`` (a run killed while writing can resume from the previous one)."""
    import pickle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(content, fh, protocol=pickle.HIGHEST_PROTOCOL)
    if path.exists():
        os.replace(path, path.with_name(path.name + ".prev"))
    os.replace(tmp, path)


def read_restart(path: Path) -> dict:
    """A restart written by :func:`write_restart`; falls back to ``<path>.prev`` if the file
    is missing or unreadable."""
    import pickle

    path = Path(path)
    errors = []
    for candidate in (path, path.with_name(path.name + ".prev")):
        try:
            with open(candidate, "rb") as fh:
                content = pickle.load(fh)
            if content.get("version") != RESTART_VERSION:
                raise ValueError(
                    f"restart format version {content.get('version')}, this code reads "
                    f"version {RESTART_VERSION} (write it again with the current code)"
                )
            return content
        except Exception as exc:  # noqa: BLE001 - report every candidate
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("no readable restart: " + "; ".join(errors))


# 2: layer arrays sized by model.yaml max_layers (Phase 5b; version 1 had 100 slots)
# 3: a leading patch axis on every state array (Phase 6 step 3, patches run batched)
RESTART_VERSION = 3


def gridcell_variables(names, level) -> tuple:
    """The variables of a stream accumulated per gridcell inside the chunk (streams with
    level ``gridcell``): named variables without a ``layer`` axis (profiles do not average
    across patches with different layer grids)."""
    from clm_ml.outputs import VARIABLES_BY_NAME

    if "gridcell" not in level:
        return ()
    return tuple(
        n for n in names if n in VARIABLES_BY_NAME and "layer" not in VARIABLES_BY_NAME[n].dims
    )


def batch_failures(errors, state, template_state, batch: PatchBatch, alive=None) -> dict:
    """``{patch: [problem, ...]}`` for the patches (of those ``alive``) that fail a
    conservation check or whose state is unhealthy."""
    errors = np.asarray(errors, dtype=float).reshape(batch.n_patches, len(BALANCE_CHECKS))
    failed = {}
    for p in range(batch.n_patches):
        if alive is not None and not alive[p]:
            continue
        problems = [
            f"{name}: error {e:.3g} exceeds {tol:g}"
            for (name, tol), e in zip(BALANCE_CHECKS, errors[p])
            if e > tol
        ]
        take = lambda t: jax.tree.map(lambda x: np.asarray(x)[p], t)  # noqa: E731
        problems += health_problems(take(state), take(template_state))
        if problems:
            failed[p] = problems
    return failed


def check_batch_balance(errors, batch: PatchBatch, where: str, alive=None) -> None:
    """Fortran's conservation checks per patch: stop the run naming the patches that fail
    (decision 25, default)."""
    errors = np.asarray(errors, dtype=float).reshape(batch.n_patches, len(BALANCE_CHECKS))
    failed = []
    for p in range(batch.n_patches):
        if alive is not None and not alive[p]:
            continue
        for (name, tol), e in zip(BALANCE_CHECKS, errors[p]):
            if e > tol:
                failed.append(f"{batch.names[p]}: {name}: error {e:.3g} exceeds {tol:g}")
    if failed:
        from clm_src_main.abortutils import endrun

        endrun(msg=f" ERROR: ({where}) " + "; ".join(failed))


def check_batch(errors, state, template_state, batch: PatchBatch, where: str) -> None:
    """Fortran's conservation checks and the state's health, per patch: stop the run naming
    the patches that fail (decision 25, default)."""
    check_batch_balance(errors, batch, where)
    for p in range(batch.n_patches):
        take = lambda t: jax.tree.map(lambda x: np.asarray(x)[p], t)  # noqa: E731
        check_health(take(state), take(template_state), f"{where}, {batch.names[p]}")


def freeze_patches(state, previous, alive):
    """``state`` with the patches not ``alive`` taken from ``previous`` (their last good
    state, so they stay finite while excluded)."""
    keep = jnp.asarray(alive)
    return jax.tree.map(
        lambda new, old: jnp.where(keep.reshape((-1,) + (1,) * (new.ndim - 1)), new, old),
        state,
        previous,
    )


def fill_patches(buffers, alive):
    """Stream buffers with the patch values of the patches not ``alive`` set to NaN (fill;
    gridcell values, ``gridcell:<variable>``, are already without them)."""
    dead = jnp.asarray(~np.asarray(alive))

    def fill(values: dict) -> dict:
        out = {}
        for k, v in values.items():
            if k.startswith("gridcell:") or v.ndim < 2:
                out[k] = v
            else:
                mask = dead.reshape((1, -1) + (1,) * (v.ndim - 2))
                out[k] = jnp.where(mask, jnp.nan, v)
        return out

    return {k: fill(v) if isinstance(v, dict) else v for k, v in buffers.items()}


def real_part(buffers, batch: PatchBatch):
    """Stream buffers without the padding patches and gridcells (:meth:`PatchBatch.padded`)."""
    if batch.n_real == batch.n_patches and batch.n_real_gridcells == batch.n_gridcells:
        return buffers

    def cut(values: dict) -> dict:
        return {
            k: v[:, : batch.n_real_gridcells if k.startswith("gridcell:") else batch.n_real]
            for k, v in values.items()
        }

    return {k: cut(v) if isinstance(v, dict) else v for k, v in buffers.items()}


def run_streams(
    batch,
    streams,
    max_layers: int,
    on_records,
    params=None,
    *,
    restart: Optional[dict] = None,
    restart_path: Optional[Path] = None,
    restart_every=None,
    save_extra=None,
    restore_extra=None,
    levels=None,
    on_patch_error: str = "stop",
    max_failed_patch_fraction: float = 0.01,
) -> None:
    """Run the model with output streams (plan §6.9): the first step step by step, then one
    compiled chunk per calendar month accumulating every stream inside the scan. After each
    chunk the state's health is checked, and a restart is written (``restart_path``) when the
    chunk ends on a ``restart_every`` boundary and at the end of the run.

    ``batch`` is a :class:`PatchBatch` (or one :class:`LegacyRun`, a batch of one patch): the
    patches are set up one by one and advanced together, the step batched over the patch
    axis (Phase 6 step 3). Output values have a patch axis, ``(patch, ...)``; streams whose
    ``levels[i]`` include ``gridcell`` also carry ``gridcell:<variable>`` values ``(gridcell,
    ...)``, the area-weighted sums over each gridcell's patches.

    ``on_records(stream_index, records)`` receives each stream's completed records
    ``[(key, end_time, {variable: value}), ...]`` as soon as they are complete.

    ``restart`` (from :func:`read_restart`) starts from a saved state: if its time is inside
    the run, the run continues from there (with the saved output bookkeeping; results equal
    an uninterrupted run); if it is the run's start, a new run starts from that state. The
    layer grid (the template) is rebuilt by running the initialization step, whose results
    are then discarded. ``save_extra()`` / ``restore_extra(saved)`` carry the caller's own
    bookkeeping (e.g. output files in progress) in the restart. ``params`` (stacked per
    patch) replaces the patches' own parameters.

    A patch that fails a conservation check or the health check (decision 25): with
    ``on_patch_error="stop"`` the run stops naming it; with ``"exclude"`` it is left out from
    then on (reported; its state frozen at its last good chunk, its outputs fill values,
    gridcell values from the other patches with renormalized weights), and the run stops
    when more than ``max_failed_patch_fraction`` of the patches have failed."""
    from clm_ml import output_streams as S

    if isinstance(batch, LegacyRun):
        batch = PatchBatch.single(batch)
    run = batch.run
    n_p, n_g = batch.n_patches, batch.n_gridcells
    levels = levels or [("patch",)] * len(streams)
    names = sorted({n for spec in streams for n in spec.variables})
    gvars = [gridcell_variables(spec.variables, lv) for spec, lv in zip(streams, levels)]
    collectors = [S.StreamCollector(spec) for spec in streams]
    step_seconds = run.time_of(0).dtime
    run_start = step_starts(run, 0, 1)[0]
    if params is None:
        params = batch.params()
    if on_patch_error not in ("stop", "exclude"):
        raise ValueError(f"on_patch_error must be stop or exclude, got {on_patch_error!r}")
    alive = np.ones(n_p, dtype=bool)

    def step_end(k):
        return step_starts(run, k, 1)[0] + datetime.timedelta(seconds=step_seconds)

    # First step of every patch (initializes the canopy and builds the template): run at
    # setup (PatchBatch.firsts), or here for a single run
    firsts = batch.firsts
    if firsts is None:
        firsts = [PatchBatch.first_step(r, g) for r, g in zip(batch.runs, batch.grids)]
    filt = firsts[0][1]
    template = stack_patches([st for st, _ in firsts])
    state = stack_patches([state_from_instances(st) for st, _ in firsts])
    template_state = state
    shapes1 = output_shapes(run, firsts[0][0], names, max_layers)  # one patch: (1, ...)
    shapes = {n: (n_p,) + sh[1:] for n, sh in shapes1.items()}

    def buffer_shapes(spec, gv):
        out = {n: shapes[n] for n in spec.variables}
        out.update({f"gridcell:{n}": (n_g,) + shapes[n][1:] for n in gv})
        return out

    def gridcell_sums(values, gv):
        out = {}
        for n in gv:
            v = np.asarray(values[n], dtype=float)
            w = batch.weight.reshape((-1,) + (1,) * (v.ndim - 1))
            g = np.zeros((n_g,) + v.shape[1:])
            np.add.at(g, batch.gridcell_index, v * w)
            out[f"gridcell:{n}"] = g
        return out

    if restart is None:
        first = 1
        per_patch = [step_values(st, names, max_layers) for st, _ in firsts]
        values = {n: np.concatenate([v[n] for v in per_patch]) for n in names}
        for i, (spec, gv) in enumerate(zip(streams, gvars)):
            rec = S.chunk_records(step_starts(run, 0, 1), 0, step_seconds, spec.frequency)
            buffers = S.init_buffers(spec.statistic, 1, buffer_shapes(spec, gv), jnp)
            vals = {n: jnp.asarray(values[n]) for n in spec.variables}
            vals.update({k: jnp.asarray(v) for k, v in gridcell_sums(values, gv).items()})
            buffers = S.accumulate(spec.statistic, buffers, 0, vals)
            on_records(i, collectors[i].add_chunk(rec, real_part(buffers, batch)))
    else:
        elapsed = (restart["time"] - run_start).total_seconds()
        first = int(round(elapsed / step_seconds))
        if first * step_seconds != elapsed or not 0 <= first < run.ntimes:
            raise ValueError(
                f"restart time {restart['time']} is not a step of this run "
                f"({run_start} + {run.ntimes} steps of {step_seconds} s)"
            )
        state = jax.tree.map(jnp.asarray, restart["state"])
        for (where, a), b in zip(
            jax.tree_util.tree_leaves_with_path(state), jax.tree.leaves(template_state)
        ):
            if np.shape(a) != np.shape(b):
                raise ValueError(
                    f"restart state {jax.tree_util.keystr(where)} has shape {np.shape(a)}, "
                    f"this run {np.shape(b)} (e.g. a different model.yaml max_layers or "
                    "number of patches)"
                )
        if first > 0:  # continuing the same run: restore the output bookkeeping
            alive = np.asarray(restart.get("alive", alive), dtype=bool)
            for collector, pending in zip(collectors, restart["collectors"]):
                collector._pending = pending
            if restore_extra is not None:
                restore_extra(restart.get("extra"))

    runner = make_batch_runner(batch, template, filt, streams, max_layers, gvars)

    # The next chunk's forcing is read on a background thread while the current chunk runs
    # (plan §6.9; jax-gcm overlaps I/O the same way).
    from concurrent.futures import ThreadPoolExecutor

    schedule = month_chunks(run, first)
    reader = ThreadPoolExecutor(max_workers=1)
    pending = reader.submit(read_batch_inputs, batch, *schedule[0]) if schedule else None
    for c, (first_step, n) in enumerate(schedule):
        xg, xp = pending.result()
        if c + 1 < len(schedule):
            pending = reader.submit(read_batch_inputs, batch, *schedule[c + 1])
        starts = step_starts(run, first_step, n)
        records = [S.chunk_records(starts, first_step, step_seconds, s.frequency) for s in streams]
        buffers = tuple(
            S.init_buffers(s.statistic, len(rc.keys), buffer_shapes(s, gv), jnp)
            for s, rc, gv in zip(streams, records, gvars)
        )
        slots = tuple(jnp.asarray(rc.index) for rc in records)
        weight = jnp.asarray(batch.weight, dtype=jnp.float64)
        gidx = jnp.asarray(batch.gridcell_index)
        chunk_start_state = state
        state, buffers, errors = runner(
            state, xg, xp, slots, buffers, params, weight, jnp.asarray(alive), gidx
        )
        end = step_end(first_step + n - 1)
        where = f"the chunk {starts[0]} to {end}"
        if on_patch_error == "stop":
            check_batch(errors, state, template_state, batch, where)
        else:
            failed = batch_failures(errors, state, template_state, batch, alive)
            for p, problems in failed.items():
                print(f"excluding {batch.names[p]} after {where}: " + "; ".join(problems))
                alive[p] = False
            n_out = int((~alive[: batch.n_real]).sum())
            if n_out > max_failed_patch_fraction * batch.n_real:
                from clm_src_main.abortutils import endrun

                endrun(
                    msg=f" ERROR: {n_out} of {batch.n_real} patches failed (more than "
                    f"max_failed_patch_fraction {max_failed_patch_fraction:g}), after {where}"
                )
            if not alive.all():
                state = freeze_patches(state, chunk_start_state, alive)
                buffers = tuple(fill_patches(b, alive) for b in buffers)
        for i, rc in enumerate(records):
            on_records(i, collectors[i].add_chunk(rc, real_part(buffers[i], batch)))
        last = first_step + n == run.ntimes
        if restart_path is not None and (last or restart_boundary(end, restart_every)):
            write_restart(
                restart_path,
                {
                    "version": RESTART_VERSION,
                    "time": end,
                    "state": jax.tree.map(np.asarray, state),
                    "collectors": [c._pending for c in collectors],
                    "alive": alive.copy(),
                    "extra": save_extra() if save_extra is not None else None,
                },
            )
    reader.shutdown()
    for i, collector in enumerate(collectors):
        on_records(i, collector.close())


def main(opts: Optional[MLOptions] = None):
    """``clm-ml-offline NAMELIST``. ``opts``: the multilayer canopy options when called by
    ``clm_ml`` (legacy text output of a ``run.yaml`` run); see :func:`setup_run`."""
    source = sys.argv[1] if len(sys.argv) > 1 else None
    run = setup_run(read_namelist(source), opts)
    params, filt, st = run.params, run.filt, run.insts
    ntimes, tower_num, iyear, imonth, dirout = (
        run.ntimes,
        run.tower_num,
        run.iyear,
        run.imonth,
        run.dirout,
    )
    from offline_driver import TowerDataMod
    from offline_driver.CLMml_driver import output

    # ------------------------------------------------------------------
    # 7. Open output files  (mirrors CLMml_drv lines 123-153)
    # ------------------------------------------------------------------
    if dirout:
        os.makedirs(dirout, exist_ok=True)
    else:
        dirout = "."

    tid = str(TowerDataMod.tower_id[tower_num])

    def _outpath(tag: str) -> str:
        fname = f"{tid}_{iyear:04d}-{imonth:02d}_{tag}.out"
        return os.path.join(dirout, fname)

    fout1 = open(_outpath("flux"), "w")
    fout2 = open(_outpath("aux"), "w")
    fout3 = open(_outpath("profile"), "w")
    fout4 = open(_outpath("fsun"), "w")
    fout5 = open(_outpath("fluxprofile"), "w")
    fout6 = open(_outpath("soiltemp"), "w")
    print(f"Output files opened in: {dirout}", flush=True)

    # ------------------------------------------------------------------
    # 8. Time loop  (mirrors CLMml_drv lines 165-196)
    # ------------------------------------------------------------------
    progress_nsteps = int(params.get("progress_nsteps", 1))
    if progress_nsteps <= 0:
        progress_nsteps = 1
    print(
        f"Starting time loop: ntimes={ntimes}, dtstep={run.time_of(0).dtime}s, "
        f"progress every {progress_nsteps} step(s)",
        flush=True,
    )

    # Steps after the first run as compiled chunks (one jax.lax.scan each, plan Phase 4):
    # one chunk per calendar month by default (decision 17); namelist chunk_steps = N gives
    # fixed N-step chunks, and 0 runs every step separately (no compilation of the step;
    # slower, for debugging).
    chunk_steps = int(params["chunk_steps"]) if "chunk_steps" in params else None

    def write_output(calday, mlcanopy, temperature):
        output(
            calday,
            tower_num,
            fout1,
            fout2,
            fout3,
            fout4,
            fout5,
            fout6,
            mlcanopy,
            temperature,
            met_type=run.opts.met_type,
        )

    def report(itim, step_dt, detail=""):
        if itim == 1 or itim == ntimes or (itim % progress_nsteps == 0) or detail:
            elapsed = time.perf_counter() - run_t0
            eta = max(ntimes - itim, 0) * elapsed / float(itim)
            print(
                f"  timestep {itim}/{ntimes} | step={step_dt:.2f}s {detail}"
                f"| elapsed={elapsed:.1f}s | eta={eta:.1f}s",
                flush=True,
            )

    run_t0 = time.perf_counter()
    try:
        # First step: initializes the canopy (step by step, not compiled). The containers
        # after it are the template every later step is rebuilt from, with the carried
        # prognostic state (clm_ml.state.State).
        step_t0 = time.perf_counter()
        inputs = step_inputs(run, run.load_chunk(0, 1), 0)
        st, filt = advance(run, st, filt, inputs, initialize=True)
        template, state = st, state_from_instances(st)
        write_output(inputs.now.calday, st.mlcanopy, st.temperature)
        report(1, time.perf_counter() - step_t0)

        if chunk_steps is None or chunk_steps > 0:
            runner = make_chunk_runner(run, template, filt)
            if chunk_steps is None:
                schedule = month_chunks(run, 1)
            else:
                schedule = [
                    (k, min(chunk_steps, ntimes - k)) for k in range(1, ntimes, chunk_steps)
                ]
            for first, n in schedule:
                chunk_t0 = time.perf_counter()
                chunk = run.load_chunk(first, n)
                inputs = [step_inputs(run, chunk, k) for k in range(first, first + n)]
                # Fixed-length chunks: a short last chunk is padded (repeating its last
                # input) to the chunk length so it reuses the compiled scan; the extra steps
                # are discarded. Monthly chunks are run at their own length (each distinct
                # length compiles once).
                pad = (chunk_steps - n) if chunk_steps else 0
                padded = inputs + [inputs[-1]] * pad
                _, (states, mlcanopy, temperature) = runner(state, stack_step_inputs(padded))
                check_balance(mlcanopy.balance_error[:n], f"time steps {first + 1}-{first + n}")
                state = jax.tree.map(lambda a: a[n - 1], states)
                for k, inp in enumerate(inputs):
                    write_output(
                        inp.now.calday,
                        jax.tree.map(lambda a, k=k: a[k], mlcanopy),
                        jax.tree.map(lambda a, k=k: a[k], temperature),
                    )
                chunk_dt = time.perf_counter() - chunk_t0
                report(first + n, chunk_dt / n, detail=f"(chunk of {n} steps: {chunk_dt:.1f}s) ")
        else:
            for _itim in range(2, ntimes + 1):
                step_t0 = time.perf_counter()
                inputs = step_inputs(run, run.load_chunk(_itim - 1, 1), _itim - 1)
                state = advance_acclimation(run, state, inputs)
                st = instances_with_state(template, state)
                st, filt = advance(run, st, filt, inputs, initialize=False)
                state = state_from_instances(st, state)
                write_output(inputs.now.calday, st.mlcanopy, st.temperature)
                report(_itim, time.perf_counter() - step_t0)
    finally:
        for fh in (fout1, fout2, fout3, fout4, fout5, fout6):
            try:
                fh.close()
            except Exception:
                pass

    print("Run complete.", flush=True)


if __name__ == "__main__":
    main()
