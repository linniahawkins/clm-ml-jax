"""
The model in this Python process: initial state, compiled chunks, gradients (plan Phase 4).

``Run.execute()`` runs the model in a separate process and writes files. :class:`Model` is
for working in a notebook or script instead: it sets the model up in *this* process and
exposes the pieces a calibration or sensitivity study needs::

    from clm_ml.run import load_run
    from clm_ml.model import Model

    model = Model.from_run(load_run("chats7/run.yaml"))
    state = model.initial_state()          # after the first (initializing) step
    params = model.default_params()        # CanopyParams table, one row per PFT of the domain
    first, n = model.chunks()[0]           # calendar-month chunks (decision 17)
    forcing = model.forcing(first, n)
    daily = StreamSpec("daily", "daily", "mean", ("latent_heat_flux",))

    def loss(p):
        new_state, records = model.run_chunk(state, forcing, first, p, streams=[daily])
        return records["daily"]["latent_heat_flux"].mean()

    jax.grad(loss)(params)                 # d(loss)/d(every parameter)

``run_chunk`` is differentiable with respect to the state, the parameters, the forcing and
the patch weights. All patches of the run (any number of gridcells and patches) advance
together, the time step batched over the patch axis (Phase 6 step 3): state and outputs have
a leading patch axis, and gridcell outputs (``gridcell:<variable>``) are the area-weighted
sums of the patches.

Options and parameters are held by each model (Phase 5a), and the per-site data the time
step reads (soil geometry, bedrock depth, soil colour, latitude and longitude) are in its
containers (Phase 6 step 2), so a process can hold several models, of the same or different
towers, e.g. to compare schemes, parameter sets or sites. The soil layer structure
(``soil.layer_structure``) is still fixed per process (array sizes chosen when the modules
are imported).
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Sequence

import numpy as np

_PREPARED: Optional[str] = None  # soil layer structure this process is set up for

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
    """A run's patches set up in this process (Phase 6 step 3: any number of gridcells and
    patches, the step batched over the patch axis). State, template and outputs have a
    leading patch axis; the forcing comes per gridcell and per patch (:meth:`forcing`); the
    parameters are a table with one row per PFT of the domain (:attr:`pft_names`), gathered
    to the patches, so their gradients are per PFT (summed over each PFT's patches)."""

    def __init__(self, run, batch, template, filt, state):
        from clm_ml.params import PFT_NAMES

        self.run = run
        self._batch = batch
        self._legacy = batch.run  # the first patch's run: clock, options, structure
        self._template = template
        self._filt = filt
        self._state = state
        self._runners: dict = {}
        pfts = [PFT_NAMES[int(i)] for i in run.domain.patches.pft_index]
        self.pft_names: tuple = tuple(dict.fromkeys(pfts))
        rows = np.array([self.pft_names.index(n) for n in pfts])
        self._pft_rows = rows[batch.source]  # the batch's order (padding: the patch copied)

    @classmethod
    def from_run(cls, run) -> "Model":
        """Set the model up in this process for ``run`` (a :class:`clm_ml.run.Run`): every
        patch, each with its first (initializing) step."""
        global _PREPARED
        spec = run.legacy_spec(None)
        key = spec["namelist"]["clm_phys"]
        if _PREPARED is not None and _PREPARED != key:
            raise RuntimeError(
                f"this process holds models with {_PREPARED} soil layers; {key} needs a new "
                "Python process: the soil layer structure fixes array sizes when the modules "
                "are imported (or use Run.execute(), which runs in a separate process)"
            )
        from clm_ml.legacy import worker

        # Each patch installs its own tables and tower values before its setup; afterwards
        # the model uses only what it holds (options, parameters, state).
        _PREPARED = key
        import offline_executable.main as M
        from clm_ml.legacy.state import state_from_instances

        batch = worker.setup_batch(spec)
        max_layers = run.config.canopy_layers.max_layers
        template, _ = M.batched_first_step(batch, (), max_layers)
        filt = batch.init_filt if batch.inits is None else batch.inits[0][1]
        return cls(run, batch, template, filt, state_from_instances(template))

    # -- pieces ------------------------------------------------------------------------

    @property
    def n_patches(self) -> int:
        return self._batch.n_patches

    def initial_state(self):
        """The :class:`clm_ml.state.State` after the first step (the start of chunk 1), with
        a leading patch axis."""
        return self._state

    def default_params(self):
        """The :class:`CanopyParams` table from the run's params.yaml: one row per PFT of
        the domain (:attr:`pft_names`), leading axis."""
        import offline_executable.main as M

        return M.stack_patches([self.run.params.canopy_params(n) for n in self.pft_names])

    def patch_params(self, params):
        """A PFT table (:meth:`default_params`) gathered to the patches (leading patch axis)."""
        import jax
        import jax.numpy as jnp

        rows = jnp.asarray(self._pft_rows)
        return jax.tree.map(lambda x: jnp.take(x, rows, axis=0), params)

    def weights(self):
        """The patches' area fractions of their gridcells, in the batch's order (domain order,
        then padding patches with 0; device blocks with ``execution.devices``)."""
        import jax.numpy as jnp

        return jnp.asarray(self._batch.weight, dtype=jnp.float64)

    def chunks(self) -> list:
        """``(first_step, n_steps)`` of the run's compiled chunks after the first step."""
        import offline_executable.main as M

        return M.month_chunks(self._legacy, 1)

    def forcing(self, first_step: int, n_steps: int):
        """The forcing of run steps ``first_step ... first_step + n_steps - 1``: ``(xg, xp)``,
        the meteorology per gridcell and the CLM history record per patch, stacked per step
        (differentiable input of :meth:`run_chunk`)."""
        import offline_executable.main as M

        return M.read_batch_inputs(self._batch, first_step, n_steps)

    # -- running -----------------------------------------------------------------------

    def _runner(self, streams, gvars):
        import offline_executable.main as M

        key = tuple((s.name, s.frequency, s.statistic, s.variables) for s in streams)
        if key not in self._runners:
            max_layers = self.run.config.canopy_layers.max_layers
            self._runners[key] = M.make_batch_runner(
                self._batch, self._template, self._filt, streams, max_layers, gvars
            )
        return self._runners[key]

    def run_chunk(
        self,
        state,
        forcing,
        first_step: int,
        params=None,
        streams: Sequence = (),
        weights=None,
        program: Optional["Model"] = None,
        alive=None,
    ):
        """Advance ``state`` over the chunk starting at run step ``first_step`` (``forcing``
        from :meth:`forcing`) as one compiled scan, accumulating ``streams``
        (:class:`clm_ml.output_streams.StreamSpec`) inside it. ``params``: a PFT table
        (:meth:`default_params`); ``weights``: the patches' area fractions (default
        :meth:`weights`). ``program``: run this model's inputs through another model's
        compiled program (same number of patches and layer slots). ``alive``: the patches
        still in the run (default all; decision 25): the others are left out of gridcell
        values and of parameter gradients.

        Returns ``(state, {stream: {variable: (record, patch, ...), "gridcell:<variable>":
        (record, gridcell, ...)}})`` with the chunk's records (partial records at the chunk's
        ends included); the gridcell values are area-weighted sums over each gridcell's
        patches, formed at every step (variables without a layer axis). Differentiable with
        respect to the state, the forcing, the parameters and the weights."""
        import jax
        import jax.numpy as jnp

        import offline_executable.main as M
        from clm_ml import output_streams as S

        streams = tuple(streams)
        params = self.default_params() if params is None else params
        weights = self.weights() if weights is None else weights
        alive = jnp.ones(self.n_patches, dtype=bool) if alive is None else jnp.asarray(alive)
        xg, xp = forcing
        n = int(xg[0].nstep.shape[0])
        n_g = self._batch.n_gridcells
        max_layers = self.run.config.canopy_layers.max_layers
        names = sorted({v for s in streams for v in s.variables})
        gvars = [M.gridcell_variables(s.variables, ("gridcell",)) for s in streams]
        runner = (program or self)._runner(streams, gvars)
        first_insts = jax.tree.map(lambda x: x[0], self._template)
        shapes = {
            v: (self.n_patches,) + sh[1:]
            for v, sh in M.output_shapes(self._legacy, first_insts, names, max_layers).items()
        }
        starts = M.step_starts(self._legacy, first_step, n)
        step_seconds = self._legacy.time_of(0).dtime
        records = [S.chunk_records(starts, first_step, step_seconds, s.frequency) for s in streams]

        def buffer_shapes(s, gv):
            out = {v: shapes[v] for v in s.variables}
            out.update({f"gridcell:{v}": (n_g,) + shapes[v][1:] for v in gv})
            return out

        buffers = tuple(
            S.init_buffers(s.statistic, len(rc.keys), buffer_shapes(s, gv), jnp)
            for s, rc, gv in zip(streams, records, gvars)
        )
        slots = tuple(jnp.asarray(rc.index) for rc in records)
        state, buffers, errors = runner(
            state,
            xg,
            xp,
            slots,
            buffers,
            self.patch_params(params),
            weights,
            alive,
            jnp.asarray(self._batch.runner_gidx),
            self._template,
        )
        from multilayer_canopy.MLMathToolsMod import is_concrete

        if is_concrete(errors):  # not under jax.grad: Fortran's conservation checks
            M.check_batch_balance(
                errors, self._batch, f"the chunk from run step {first_step}", np.asarray(alive)
            )
        out = {}
        for s, b in zip(streams, (M.real_part(b, self._batch) for b in buffers)):
            if s.statistic == "mean":
                count = b["count"]
                out[s.name] = {
                    v: x / count.reshape((-1,) + (1,) * (x.ndim - 1)) for v, x in b["sum"].items()
                }
            else:
                out[s.name] = dict(b["value"])
        return state, out
