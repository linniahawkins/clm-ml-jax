"""Each step's time (clm_ml.calendar.step_time) agrees with an independent datetime
calculation."""

import datetime as dt

import pytest


def _datetime_calday(when: dt.datetime) -> float:
    """Day of year + fraction of day (1.000 = 0Z on 1 January)."""
    return when.timetuple().tm_yday + (when.hour * 3600 + when.minute * 60) / 86400.0


def test_step_time_matches_datetime():
    from clm_ml.calendar import step_time

    start = dt.datetime(2007, 5, 1)
    for n in range(1, 31 * 48 + 1):
        end = start + dt.timedelta(seconds=n * 1800)
        st = step_time(start, 1800, n, "GREGORIAN")
        assert (st.year, st.month, st.day) == (end.year, end.month, end.day)
        assert st.calday == pytest.approx(_datetime_calday(end), abs=1e-12)
