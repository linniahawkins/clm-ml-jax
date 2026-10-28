"""Monthly compiled chunks (plan decision 17): a chunk is the steps starting in one month."""

import types

from clm_src_utils.clm_time_manager import step_time
from offline_executable.main import month_chunks


def _run(start_ymd, ntimes, dtime=1800):
    return types.SimpleNamespace(
        ntimes=ntimes, time_of=lambda k: step_time(start_ymd, 0, dtime, k + 1)
    )


def test_split_at_month_start():
    # 30 April 00:00, 3 days of half-hourly steps: steps 0-47 start in April.
    run = _run(20070430, 3 * 48)
    assert month_chunks(run, 1) == [(1, 47), (48, 96)]


def test_run_within_one_month_is_one_chunk():
    run = _run(20070501, 50)
    assert month_chunks(run, 1) == [(1, 49)]


def test_step_ending_at_midnight_belongs_to_the_month_it_starts_in():
    # The step from 31 May 23:30 to 1 June 00:00 starts in May.
    run = _run(20070531, 49)
    assert month_chunks(run, 0) == [(0, 48), (48, 1)]
