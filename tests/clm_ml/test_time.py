"""Pure calendar functions in clm_time_manager agree with the global-state functions and
with an independent datetime calculation."""

import datetime as dt

import pytest


@pytest.fixture
def tm():
    from clm_src_utils import clm_time_manager

    saved = {
        k: getattr(clm_time_manager, k)
        for k in ("itim", "dtstep", "start_date_ymd", "start_date_tod")
    }
    yield clm_time_manager
    for k, v in saved.items():
        setattr(clm_time_manager, k, v)


def _datetime_calday(when: dt.datetime) -> float:
    """Day of year + fraction of day (1.000 = 0Z on 1 January)."""
    return when.timetuple().tm_yday + (when.hour * 3600 + when.minute * 60) / 86400.0


@pytest.mark.parametrize(
    "start_ymd, start_tod, dtime, nsteps",
    [
        (20070501, 0, 1800, 31 * 48),  # CHATS7 May 2007
        (20080225, 0, 1800, 10 * 48),  # across 29 February (leap year)
        (20071230, 43200, 3600, 72),  # across the new year
    ],
)
def test_step_time_matches_global_functions(tm, start_ymd, start_tod, dtime, nsteps):
    tm.start_date_ymd, tm.start_date_tod, tm.dtstep = start_ymd, start_tod, dtime
    for n in range(1, nsteps + 1):
        tm.itim = n
        st = tm.step_time(start_ymd, start_tod, dtime, n)
        assert (st.year, st.month, st.day, st.tod) == tm.get_curr_date()
        assert st.calday == tm.get_curr_calday(offset=0)
        assert st.calday_beg == tm.get_curr_calday(offset=-dtime)
        assert (st.nstep, st.dtime) == (tm.get_nstep(), tm.get_step_size())


def test_step_time_matches_datetime():
    from clm_src_utils.clm_time_manager import step_time

    start = dt.datetime(2007, 5, 1)
    for n in range(1, 31 * 48 + 1):
        end = start + dt.timedelta(seconds=n * 1800)
        st = step_time(20070501, 0, 1800, n)
        assert (st.year, st.month, st.day) == (end.year, end.month, end.day)
        assert st.calday == pytest.approx(_datetime_calday(end), abs=1e-12)
