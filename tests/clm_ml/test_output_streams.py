"""Output streams: records, accumulation in a compiled scan, and records spanning chunks."""

import datetime as dt

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clm_ml import output_streams as S

jax.config.update("jax_enable_x64", True)

STEP = 1800
START = dt.datetime(2007, 5, 1, 0, 0)


def _starts(n, first=0):
    return [START + dt.timedelta(seconds=STEP * (first + k)) for k in range(n)]


def _run_chunk(spec, first, n, value_of):
    """Scan over steps first..first+n-1 accumulating x = value_of(step); returns buffers."""
    rec = S.chunk_records(_starts(n, first), first, STEP, spec.frequency)
    shapes = {"x": (2,)}
    buffers = S.init_buffers(spec.statistic, len(rec.keys), shapes, jnp)

    def body(buf, inp):
        k, slot = inp
        return S.accumulate(spec.statistic, buf, slot, {"x": value_of(k)}), None

    steps = jnp.arange(first, first + n, dtype=float)
    buffers, _ = jax.jit(lambda b: jax.lax.scan(body, b, (steps, jnp.asarray(rec.index))))(buffers)
    return rec, buffers


def _collect(spec, chunks, value_of):
    collector = S.StreamCollector(spec)
    out = []
    for first, n in chunks:
        out += collector.add_chunk(*_run_chunk(spec, first, n, value_of))
    return out + collector.close()


def value_of(k):
    return jnp.stack([k, -2.0 * k])


@pytest.mark.parametrize("statistic", S.STATISTICS)
def test_daily_records_match_numpy_and_do_not_depend_on_chunks(statistic):
    spec = S.StreamSpec("d", "daily", statistic, ("x",))
    one = _collect(spec, [(0, 144)], value_of)  # 3 days in one chunk
    split = _collect(spec, [(0, 30), (30, 70), (100, 44)], value_of)  # chunks split mid-day
    assert [r[0] for r in one] == [dt.date(2007, 5, d) for d in (1, 2, 3)]
    for (k1, e1, v1), (k2, e2, v2) in zip(one, split):
        assert k1 == k2 and e1 == e2
        np.testing.assert_allclose(v1["x"], v2["x"], rtol=1e-15)
    k = np.arange(144, dtype=float).reshape(3, 48)
    x = np.stack([k, -2.0 * k], axis=-1)  # (day, step, component)
    reduce = {
        "mean": lambda a: a.mean(1),
        "min": lambda a: a.min(1),
        "max": lambda a: a.max(1),
        "instantaneous": lambda a: a[:, -1],
    }[statistic]
    np.testing.assert_allclose(np.stack([v["x"] for _, _, v in one]), reduce(x))
    assert one[0][1] == dt.datetime(2007, 5, 2, 0, 0)  # record time = end of the interval


def test_step_and_hourly_records():
    hourly = _collect(S.StreamSpec("h", "hourly", "mean", ("x",)), [(0, 5)], value_of)
    assert [v["x"][0] for _, _, v in hourly] == [0.5, 2.5, 4.0]
    step = _collect(S.StreamSpec("s", "step", "instantaneous", ("x",)), [(0, 3)], value_of)
    assert [v["x"][0] for _, _, v in step] == [0.0, 1.0, 2.0]


def test_bad_spec():
    with pytest.raises(ValueError):
        S.StreamSpec("bad", "weekly", "mean", ("x",))
