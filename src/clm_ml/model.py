"""
The model in this Python process: initial state, compiled chunks, gradients (plan Phase 4).

``Run.execute()`` runs the model in a separate process and writes files. :class:`Model` is
for working in a notebook or script instead: it sets the model up in *this* process and
exposes the pieces a calibration or sensitivity study needs::

    from clm_ml.run import load_run
    from clm_ml.model import Model

    model = Model.from_run(load_run("chats7/run.yaml"))
    state = model.initial_state()          # after the first (initializing) step
    params = model.default_params()        # the patch's CanopyParams (params.yaml values)
    first, n = model.chunks()[0]           # calendar-month chunks (decision 17)
    forcing = model.forcing(first, n)
    daily = StreamSpec("daily", "daily", "mean", ("latent_heat_flux",))

    def loss(p):
        new_state, records = model.run_chunk(state, forcing, first, p, streams=[daily])
        return records["daily"]["latent_heat_flux"].mean()

    jax.grad(loss)(params)                 # d(loss)/d(every parameter)

``run_chunk`` is differentiable with respect to the state, the parameters and the forcing.

Options and parameters are held by each model (Phase 5a), so a process can hold several
models of the **same tower**, e.g. to compare schemes or parameter sets. The CLM subgrid
structure (soil layers, latitude and longitude) is still set up per tower in module state,
so models of different towers need separate processes until Phase 6 (and so do soil layer
structures: ``soil.layer_structure``). One patch per model for now (Phase 6 batches
patches).
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Sequence

_PREPARED: Optional[tuple] = None  # (tower, soil layer structure) this process is set up for

# Gradients over a compiled chunk crash (segmentation fault) in XLA's newer CPU runtime
# (the "thunk" runtime, default in JAX 0.6.2): reproduced with a 2-step chunk; the older
# runtime computes them, with or without jax.checkpoint on the step. XLA reads this flag
# when it starts, so it is set here, on import, unless already chosen. (Normal runs,
# Run.execute and the command line, keep the default runtime.) JAX 0.9.2 no longer supports
# the flag (the new runtime is the only one) and computes these gradients without crashing
# (Phase 6, step 0c), so it is set only for JAX < 0.7.
_LEGACY_CPU_RUNTIME = "--xla_cpu_use_thunk_runtime=false"


def _jax_older_than(major: int, minor: int) -> bool:
    from importlib.metadata import version

    return tuple(int(x) for x in version("jax").split(".")[:2]) < (major, minor)


if _jax_older_than(0, 7) and "xla_cpu_use_thunk_runtime" not in os.environ.get("XLA_FLAGS", ""):
    import jax._src.xla_bridge as _xla_bridge

    if _xla_bridge.backends_are_initialized():
        warnings.warn(
            "clm_ml.model: JAX has already started, so XLA's older CPU runtime cannot be "
            f"selected; gradients over chunks may crash. Import clm_ml.model first, or set "
            f"XLA_FLAGS={_LEGACY_CPU_RUNTIME} before starting Python.",
            stacklevel=2,
        )
    else:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + " " + _LEGACY_CPU_RUNTIME
        ).strip()


class Model:
    def __init__(self, run, legacy_run, template, filt, state):
        self.run = run
        self._legacy = legacy_run
        self._template = template
        self._filt = filt
        self._state = state
        self._runners: dict = {}

    @classmethod
    def from_run(cls, run) -> "Model":
        """Set the model up in this process for ``run`` (a :class:`clm_ml.run.Run`), and run
        its first (initializing) step."""
        global _PREPARED
        spec = run.legacy_spec(None)
        key = (spec["tower_name"], spec["namelist"]["clm_phys"])
        if _PREPARED is not None and _PREPARED != key:
            raise RuntimeError(
                f"this process holds models of tower {_PREPARED[0]} ({_PREPARED[1]} soil "
                f"layers); {key[0]} ({key[1]}) needs a new Python process: the CLM subgrid "
                "structure is set up per tower in module state until Phase 6 (or use "
                "Run.execute(), which runs in a separate process)"
            )
        from clm_ml.legacy import worker

        # Each model installs its own tables and tower values before its setup; afterwards
        # it uses only what it holds (options, parameters, state).
        worker.prepare_process(spec)
        _PREPARED = key
        import offline_executable.main as M
        from clm_ml.legacy.state import state_from_instances

        legacy_run = worker.setup_legacy_run(spec)
        inputs = M.step_inputs(legacy_run, legacy_run.load_chunk(0, 1), 0)
        st, filt = M.advance(legacy_run, legacy_run.insts, legacy_run.filt, inputs, True)
        return cls(run, legacy_run, st, filt, state_from_instances(st))

    # -- pieces ------------------------------------------------------------------------

    def initial_state(self):
        """The :class:`clm_ml.state.State` after the first step (the start of chunk 1)."""
        return self._state

    def default_params(self):
        """The patch's :class:`CanopyParams`, from the run's params.yaml."""
        from clm_ml.params import PFT_NAMES

        pft = PFT_NAMES[int(self.run.domain.patches.pft_index[0])]
        return self.run.params.canopy_params(pft)

    def chunks(self) -> list:
        """``(first_step, n_steps)`` of the run's compiled chunks after the first step."""
        import offline_executable.main as M

        return M.month_chunks(self._legacy, 1)

    def forcing(self, first_step: int, n_steps: int):
        """The forcing of run steps ``first_step ... first_step + n_steps - 1`` (stacked per
        step; differentiable input of :meth:`run_chunk`)."""
        import offline_executable.main as M

        chunk = self._legacy.load_chunk(first_step, n_steps)
        return M.stack_step_inputs(
            [M.step_inputs(self._legacy, chunk, k) for k in range(first_step, first_step + n_steps)]
        )

    # -- running -----------------------------------------------------------------------

    def run_chunk(self, state, forcing, first_step: int, params=None, streams: Sequence = ()):
        """Advance ``state`` over the chunk starting at run step ``first_step`` (``forcing``
        from :meth:`forcing`) as one compiled scan, accumulating ``streams``
        (:class:`clm_ml.output_streams.StreamSpec`) inside it.

        Returns ``(state, {stream: {variable: (record, patch, ...)}})`` with the chunk's
        records (partial records at the chunk's ends included). Differentiable."""
        import jax.numpy as jnp

        import offline_executable.main as M
        from clm_ml import output_streams as S

        streams = tuple(streams)
        params = self.default_params() if params is None else params
        n = int(forcing[0].nstep.shape[0])
        max_layers = self.run.config.canopy_layers.max_layers
        names = sorted({v for s in streams for v in s.variables})
        key = tuple((s.name, s.frequency, s.statistic, s.variables) for s in streams)
        if key not in self._runners:
            self._runners[key] = M.make_stream_runner(
                self._legacy, self._template, self._filt, streams, max_layers
            )
        shapes = M.output_shapes(self._legacy, self._template, names, max_layers)
        starts = M.step_starts(self._legacy, first_step, n)
        step_seconds = self._legacy.time_of(0).dtime
        records = [S.chunk_records(starts, first_step, step_seconds, s.frequency) for s in streams]
        buffers = tuple(
            S.init_buffers(s.statistic, len(rc.keys), {v: shapes[v] for v in s.variables}, jnp)
            for s, rc in zip(streams, records)
        )
        slots = tuple(jnp.asarray(rc.index) for rc in records)
        state, buffers, errors = self._runners[key](state, forcing, slots, buffers, params)
        from multilayer_canopy.MLCanopyFluxesType import check_balance
        from multilayer_canopy.MLMathToolsMod import is_concrete

        if is_concrete(errors):  # not under jax.grad: Fortran's conservation checks
            check_balance(errors, f"the chunk from run step {first_step}")
        out = {}
        for s, b in zip(streams, buffers):
            if s.statistic == "mean":
                count = b["count"]
                out[s.name] = {
                    v: x / count.reshape((-1,) + (1,) * (x.ndim - 1)) for v, x in b["sum"].items()
                }
            else:
                out[s.name] = dict(b["value"])
        return state, out
