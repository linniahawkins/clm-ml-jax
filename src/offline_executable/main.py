"""
main.py  —  JAX equivalent of prgm.exe (offline_executable).

Run with:
    python main.py < nl.CHATS7.05.2007
or:
    python main.py nl.CHATS7.05.2007

Mirrors the Fortran entry point in offline_driver/CLMml.F90 and
the namelist reader in offline_driver/controlMod.F90.
"""

import datetime
import os
import sys
import time
from pathlib import Path

import f90nml  # pip install f90nml
import jax

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


def main():
    # ------------------------------------------------------------------
    # 1. Parse namelist  (mirrors controlMod.F90 :: read(5, nml=...))
    # ------------------------------------------------------------------
    source = sys.argv[1] if len(sys.argv) > 1 else None
    nml = read_namelist(source)

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

    # setFilters returns a new NamedTuple — capture it and update every
    # module that holds a direct binding to the old filter object.
    _new_filter = _filterMod.setFilters(_filterMod.filter)
    _filterMod.filter = _new_filter

    # clm_driver.py imports filter at module level via
    # "from clm_src_main.filterMod import filter", so we must patch
    # that binding too; otherwise clm_drv() sees the stale zero-count filter.
    import clm_src_main.clm_driver as _clm_driver_mod

    _clm_driver_mod.filter = _new_filter

    print("Initialization complete.", flush=True)

    # ------------------------------------------------------------------
    # 6. Pre-loop setup  (mirrors CLMml_drv lines 84-120)
    # ------------------------------------------------------------------
    from clm_ml.legacy.forcing import (
        atm_record_current,
        atm_record_next,
        clm_record_at,
        load_forcing,
        tower_met_at,
    )
    from offline_driver.CLMml_driver import (
        SoilInit,
        TowerVeg,
        init_acclim,
        output,
    )
    from offline_driver.TowerMetMod import TowerMetCurr, tower_first_record

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

    # 6-load. Forcing for the run, read before the time loop (no file reads in the loop).
    #     The run is one chunk here; the chunk loop of plan §6.9 comes in Phase 4.
    run_start = datetime.datetime.strptime(str(start_ymd), "%Y%m%d") + datetime.timedelta(
        seconds=run_start_tod
    )
    forcing = load_forcing(
        fin1,
        fin2,
        fin_soil_adjust,
        first_record=tower_first_record(fin1, run_start, dtstep_sec),
        run_n_steps=ntimes,
        first_step=0,
        n_steps=ntimes,
        history_index=_history_index,
        tower_height=float(TowerDataMod.tower_ht[tower_num]),
    )

    # 6a. Acclimation temperature: mean over the whole run (Fortran CLM-ml v2 definition,
    #     which needs every step's forcing up front; plan Phase 4b replaces it with a
    #     10-day running mean). Array index t + 1 skips the chunk's leading halo record.
    (
        clm_instMod.atm2lnd_inst,
        clm_instMod.wateratm2lndbulk_inst,
        clm_instMod.temperature_inst,
        clm_instMod.frictionvel_inst,
        clm_instMod.mlcanopy_inst,
    ) = init_acclim(
        [tower_met_at(forcing, t + 1) for t in range(ntimes)],
        tower_num,
        ntimes,
        bounds.begp,
        bounds.endp,
        clm_instMod.atm2lnd_inst,
        clm_instMod.wateratm2lndbulk_inst,
        clm_instMod.temperature_inst,
        clm_instMod.frictionvel_inst,
        clm_instMod.mlcanopy_inst,
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
    clm_instMod.canopystate_inst, clm_instMod.mlcanopy_inst = TowerVeg(
        tower_num,
        bounds.begp,
        bounds.endp,
        clm_instMod.canopystate_inst,
        clm_instMod.mlcanopy_inst,
    )

    # 6d. Soil temperature and moisture initialisation
    clm_instMod.waterstatebulk_inst, clm_instMod.temperature_inst = SoilInit(
        fin2,
        soil_init_time_indx,
        bounds.begc,
        bounds.endc,
        clm_instMod.soilstate_inst,
        clm_instMod.waterstatebulk_inst,
        clm_instMod.temperature_inst,
    )

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
        f"Starting time loop: ntimes={ntimes}, dtstep={dtstep_sec}s, "
        f"progress every {progress_nsteps} step(s)",
        flush=True,
    )

    run_t0 = time.perf_counter()
    try:
        for _itim in range(1, ntimes + 1):
            step_t0 = time.perf_counter()
            now = _time(_itim - 1)
            curr_calday = now.calday
            t_prev, t_curr, t_next = forcing.window_indices(_itim - 1)

            # Install current-step tower meteorology
            t_met0 = time.perf_counter()
            (
                clm_instMod.atm2lnd_inst,
                clm_instMod.wateratm2lndbulk_inst,
                clm_instMod.frictionvel_inst,
            ) = TowerMetCurr(
                tower_met_at(forcing, t_curr),
                tower_num,
                bounds.begp,
                bounds.endp,
                clm_instMod.atm2lnd_inst,
                clm_instMod.wateratm2lndbulk_inst,
                clm_instMod.frictionvel_inst,
            )

            # Forcing at the previous and next steps, for the multilayer canopy's
            # 3-point time interpolation (Fortran *_bef/_next_forcing; TowerMetNext)
            n_patch = bounds.endp + 1
            forcing_bef = atm_record_current(forcing, t_prev, n_patch)
            forcing_next = atm_record_next(forcing, t_next, n_patch)
            t_met = time.perf_counter() - t_met0

            # Advance physics
            t_adv0 = time.perf_counter()
            ModelAdvance(bounds, clm_record_at(forcing, t_curr), now, forcing_bef, forcing_next)
            t_adv = time.perf_counter() - t_adv0

            # Write output
            t_out0 = time.perf_counter()
            output(
                curr_calday,
                tower_num,
                fout1,
                fout2,
                fout3,
                fout4,
                fout5,
                fout6,
                clm_instMod.mlcanopy_inst,
                clm_instMod.temperature_inst,
            )
            t_out = time.perf_counter() - t_out0

            step_dt = time.perf_counter() - step_t0
            if _itim == 1 or _itim == ntimes or (_itim % progress_nsteps == 0):
                elapsed = time.perf_counter() - run_t0
                avg_step = elapsed / float(_itim)
                eta = max(ntimes - _itim, 0) * avg_step
                print(
                    f"  timestep {_itim}/{ntimes} | step={step_dt:.2f}s "
                    f"(met={t_met:.2f}s, adv={t_adv:.2f}s, out={t_out:.2f}s) "
                    f"| elapsed={elapsed:.1f}s | eta={eta:.1f}s",
                    flush=True,
                )
    finally:
        for fh in (fout1, fout2, fout3, fout4, fout5, fout6):
            try:
                fh.close()
            except Exception:
                pass

    print("Run complete.", flush=True)


if __name__ == "__main__":
    main()
