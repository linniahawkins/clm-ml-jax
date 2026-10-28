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
jax.config.update("jax_persistent_cache_min_compile_time_secs", 10.0)

from clm_ml.legacy.state import instances_with_state, state_from_instances  # noqa: E402
from clm_src_cpl.lnd_comp_nuopc import InitializeRealize, ModelAdvance  # noqa: E402
from clm_src_main.decompMod import bounds_type  # noqa: E402
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


def setup_run(nml: dict) -> LegacyRun:
    """Configure the legacy modules from a namelist, initialize CLM, read the run's forcing
    and install initial conditions (CLMml_drv up to its time loop)."""
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
    fin_clm = _resolve_path(str(params.get("fin_clm", "")))
    fin_soil_adjust = _resolve_path(str(params.get("fin_soil_adjust", "")))
    dirout_raw = str(params.get("dirout", "")).strip()
    dirout = _resolve_path(dirout_raw) if dirout_raw else ""

    # fin1 = tower forcing file (used by TowerMet* and init_acclim)
    # fin2 = CLM history file   (used by clmData inside clm_driver)
    fin1 = fin_tower
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
    MLclm_varctl.met_type = int(params.get("met_type", MLclm_varctl.met_type))
    MLclm_varctl.dpai_min = float(params.get("dpai_min", MLclm_varctl.dpai_min))
    MLclm_varctl.pftcon_val = int(params.get("pftcon_val", MLclm_varctl.pftcon_val))

    clm_time_manager.start_date_ymd = start_ymd
    clm_time_manager.start_date_tod = int(params.get("start_tod", 0))
    clm_time_manager.dtstep = int(TowerDataMod.tower_time[tower_num]) * 60

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

    InitializeRealize(bounds)

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
    from offline_driver.TowerMetMod import tower_first_record

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

    start_calday_clm = step_time(clm_start_ymd, clm_start_tod, dtstep_sec, 1).calday

    def _history_index(run_step: int) -> int:
        """1-based CLM history time slice of a 0-based run step."""
        calday = _time(run_step).calday
        return round((calday - start_calday_clm) * 86400.0 / float(dtstep_sec)) + 1

    soil_init_time_indx = _history_index(0)

    # 6-load. Forcing is read one chunk at a time (with one halo record at each end), so
    #     memory does not depend on run length (plan §6.9).
    run_start = datetime.datetime.strptime(str(start_ymd), "%Y%m%d") + datetime.timedelta(
        seconds=run_start_tod
    )
    first_record = tower_first_record(fin1, run_start, dtstep_sec)
    tower_height = float(TowerDataMod.tower_ht[tower_num])

    from clm_ml.legacy.forcing import TowerForcingSource

    source = TowerForcingSource(
        tower_file=fin1,
        clm_history_file=fin2,
        soil_adjust_file=fin_soil_adjust,
        first_record=first_record,
        n_steps=ntimes,
        history_index=_history_index,
        tower_height=tower_height,
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

    # 6a. Acclimation temperature: mean over the whole run (Fortran CLM-ml v2 definition;
    #     plan Phase 4b replaces it with a 10-day running mean), summed over the run's
    #     forcing read chunk by chunk.
    from offline_driver.TowerMetMod import set_tower_location

    set_tower_location(tower_num, bounds.begp, bounds.endp)
    atm2lnd, wateratm2lndbulk, temperature, frictionvel, mlcanopy = init_acclim(
        run_tower_met(),
        tower_num,
        ntimes,
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

    return LegacyRun(
        params=params,
        bounds=bounds,
        filt=filt,
        insts=st,
        load_chunk=load_chunk,
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


def advance(run: LegacyRun, insts, filt, inputs: StepInputs, initialize: bool, params=None):
    """One time step: install the step's tower meteorology, then ModelAdvance.
    Returns ``(insts, filt)``."""
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
    return ModelAdvance(
        run.bounds,
        insts,
        filt,
        inputs.clm_record,
        inputs.now,
        inputs.forcing_bef,
        inputs.forcing_next,
        initialize=initialize,
        params=params,
    )


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

    def body(state, x, params):
        now, met, clm_record, forcing_bef, forcing_next = x
        inputs = StepInputs(
            now=now._replace(dtime=dtime),
            met=met,
            clm_record=clm_record,
            forcing_bef=forcing_bef,
            forcing_next=forcing_next,
        )
        insts, _ = advance(
            run,
            instances_with_state(template, state),
            filt,
            inputs,
            initialize=False,
            params=params,
        )
        state = state_from_instances(insts)
        if outputs == "state":
            return state, state
        return state, (state, insts.mlcanopy, insts.temperature)

    from multilayer_canopy.MLCanopyParamsMod import patch_setup

    _, default_params = patch_setup(int(np.asarray(filt.exposedvegp)[0]), template.mlcanopy)

    def run_chunk(state, xs, params=default_params):
        return jax.lax.scan(lambda s, x: body(s, x, params), state, xs)

    return jax.jit(run_chunk)


def make_stream_runner(run: LegacyRun, template, filt, streams, max_layers: int):
    """A compiled function ``(state, xs, slots, buffers[, params]) -> (state, buffers)``
    advancing a chunk as one ``jax.lax.scan`` and accumulating output streams inside it
    (plan §6.9): each step's output variables (``clm_ml.outputs``, computed with jax.numpy)
    are added into the slot of the stream record the step belongs to. ``streams`` are
    ``clm_ml.output_streams.StreamSpec``; ``slots[i]`` is the per-step record index of
    stream ``i`` (``chunk_records``), ``buffers[i]`` its ``init_buffers``. No per-step
    values leave the device."""
    from clm_ml import output_streams as S
    from clm_ml.outputs import VARIABLES_BY_NAME, _Raw
    from clm_src_main.clm_varpar import nlevgrnd
    from multilayer_canopy.MLCanopyParamsMod import patch_setup

    dtime = run.time_of(0).dtime
    names = sorted({n for spec in streams for n in spec.variables})
    _, default_params = patch_setup(int(np.asarray(filt.exposedvegp)[0]), template.mlcanopy)

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

    def step(carry, x):
        state, buffers = carry
        (now, met, clm_record, forcing_bef, forcing_next), slots, params = x
        inputs = StepInputs(
            now=now._replace(dtime=dtime),
            met=met,
            clm_record=clm_record,
            forcing_bef=forcing_bef,
            forcing_next=forcing_next,
        )
        insts, _ = advance(
            run,
            instances_with_state(template, state),
            filt,
            inputs,
            initialize=False,
            params=params,
        )
        values = variables(insts)
        buffers = tuple(
            S.accumulate(spec.statistic, b, slot, {n: values[n] for n in spec.variables})
            for spec, b, slot in zip(streams, buffers, slots)
        )
        return (state_from_instances(insts), buffers), None

    # As in jax-gcm: jax.checkpoint on the step, so a gradient over a chunk stores only the
    # carry (state and stream buffers) at each step and recomputes the step's intermediate
    # values in the backward pass. No effect on forward results.
    checkpointed_step = jax.checkpoint(step)

    def run_chunk(state, xs, slots, buffers, params=default_params):
        n = jax.tree.leaves(xs)[0].shape[0]
        per_step_params = jax.tree.map(lambda p: jnp.broadcast_to(p, (n,) + p.shape), params)
        (state, buffers), _ = jax.lax.scan(
            checkpointed_step, (state, buffers), (xs, slots, per_step_params)
        )
        return state, buffers

    return jax.jit(run_chunk)


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
    if problems:
        raise RuntimeError(f"unhealthy model state after {where}: " + "; ".join(problems))


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
                raise ValueError(f"restart version {content.get('version')}")
            return content
        except Exception as exc:  # noqa: BLE001 - report every candidate
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("no readable restart: " + "; ".join(errors))


RESTART_VERSION = 1


def run_streams(
    run: LegacyRun,
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
) -> None:
    """Run the model with output streams (plan §6.9): the first step step by step, then one
    compiled chunk per calendar month accumulating every stream inside the scan. After each
    chunk the state's health is checked, and a restart is written (``restart_path``) when the
    chunk ends on a ``restart_every`` boundary and at the end of the run.

    ``on_records(stream_index, records)`` receives each stream's completed records
    ``[(key, end_time, {variable: value}), ...]`` as soon as they are complete.

    ``restart`` (from :func:`read_restart`) starts from a saved state: if its time is inside
    the run, the run continues from there (with the saved output bookkeeping; results equal
    an uninterrupted run); if it is the run's start, a new run starts from that state. The
    layer grid (the template) is rebuilt by running the initialization step, whose results
    are then discarded. ``save_extra()`` / ``restore_extra(saved)`` carry the caller's own
    bookkeeping (e.g. output files in progress) in the restart."""
    from clm_ml import output_streams as S

    names = sorted({n for spec in streams for n in spec.variables})
    collectors = [S.StreamCollector(spec) for spec in streams]
    step_seconds = run.time_of(0).dtime
    run_start = step_starts(run, 0, 1)[0]

    def step_end(k):
        return step_starts(run, k, 1)[0] + datetime.timedelta(seconds=step_seconds)

    # First step (initializes the canopy and builds the template)
    st, filt = advance(
        run, run.insts, run.filt, step_inputs(run, run.load_chunk(0, 1), 0), True, params
    )
    template, state = st, state_from_instances(st)
    template_state = state
    shapes = output_shapes(run, template, names, max_layers)

    if restart is None:
        first = 1
        values = step_values(st, names, max_layers)
        for i, spec in enumerate(streams):
            rec = S.chunk_records(step_starts(run, 0, 1), 0, step_seconds, spec.frequency)
            buffers = S.init_buffers(spec.statistic, 1, {n: shapes[n] for n in spec.variables}, jnp)
            buffers = S.accumulate(
                spec.statistic, buffers, 0, {n: jnp.asarray(values[n]) for n in spec.variables}
            )
            on_records(i, collectors[i].add_chunk(rec, buffers))
    else:
        elapsed = (restart["time"] - run_start).total_seconds()
        first = int(round(elapsed / step_seconds))
        if first * step_seconds != elapsed or not 0 <= first < run.ntimes:
            raise ValueError(
                f"restart time {restart['time']} is not a step of this run "
                f"({run_start} + {run.ntimes} steps of {step_seconds} s)"
            )
        state = jax.tree.map(jnp.asarray, restart["state"])
        if first > 0:  # continuing the same run: restore the output bookkeeping
            for collector, pending in zip(collectors, restart["collectors"]):
                collector._pending = pending
            if restore_extra is not None:
                restore_extra(restart.get("extra"))

    runner = make_stream_runner(run, template, filt, streams, max_layers)

    def read_inputs(first_step, n):
        chunk = run.load_chunk(first_step, n)
        return stack_step_inputs(
            [step_inputs(run, chunk, k) for k in range(first_step, first_step + n)]
        )

    # The next chunk's forcing is read on a background thread while the current chunk runs
    # (plan §6.9; jax-gcm overlaps I/O the same way).
    from concurrent.futures import ThreadPoolExecutor

    schedule = month_chunks(run, first)
    reader = ThreadPoolExecutor(max_workers=1)
    pending = reader.submit(read_inputs, *schedule[0]) if schedule else None
    for c, (first_step, n) in enumerate(schedule):
        xs = pending.result()
        if c + 1 < len(schedule):
            pending = reader.submit(read_inputs, *schedule[c + 1])
        starts = step_starts(run, first_step, n)
        records = [S.chunk_records(starts, first_step, step_seconds, s.frequency) for s in streams]
        buffers = tuple(
            S.init_buffers(s.statistic, len(rc.keys), {v: shapes[v] for v in s.variables}, jnp)
            for s, rc in zip(streams, records)
        )
        slots = tuple(jnp.asarray(rc.index) for rc in records)
        if params is None:
            state, buffers = runner(state, xs, slots, buffers)
        else:
            state, buffers = runner(state, xs, slots, buffers, params)
        end = step_end(first_step + n - 1)
        check_health(state, template_state, f"the chunk {starts[0]} to {end}")
        for i, rc in enumerate(records):
            on_records(i, collectors[i].add_chunk(rc, buffers[i]))
        last = first_step + n == run.ntimes
        if restart_path is not None and (last or restart_boundary(end, restart_every)):
            write_restart(
                restart_path,
                {
                    "version": RESTART_VERSION,
                    "time": end,
                    "state": jax.tree.map(np.asarray, state),
                    "collectors": [c._pending for c in collectors],
                    "extra": save_extra() if save_extra is not None else None,
                },
            )
    reader.shutdown()
    for i, collector in enumerate(collectors):
        on_records(i, collector.close())


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else None
    run = setup_run(read_namelist(source))
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
        output(calday, tower_num, fout1, fout2, fout3, fout4, fout5, fout6, mlcanopy, temperature)

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
                st = instances_with_state(template, state)
                st, filt = advance(run, st, filt, inputs, initialize=False)
                state = state_from_instances(st)
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
