#!/usr/bin/env python3
"""Sun-position and timezone helpers, built on ephem and timezonefinder."""

from datetime import UTC, datetime, time, timedelta, timezone
from functools import cache
from zoneinfo import ZoneInfo

import ephem
import pandas as pd
from timezonefinder import TimezoneFinder


@cache
def _timezone_finder() -> TimezoneFinder:
    """Shared TimezoneFinder, built on first use (loading it takes ~0.4 s)."""
    return TimezoneFinder()


class SunTime:
    """Sunrise/sunset time calculator for a fixed observer location."""

    FUNC_MAP = {
        ("rise", "next"): ephem.Observer.next_rising,
        ("rise", "last"): ephem.Observer.previous_rising,
        ("set", "next"): ephem.Observer.next_setting,
        ("set", "last"): ephem.Observer.previous_setting,
    }

    def __init__(self, lat: float, lon: float, elev: float) -> None:
        """Set up an ephem.Observer at (lat, lon, elev) and resolve its timezone."""
        self.lat = lat
        self.lon = lon
        self.elev = elev

        self.tz_name = get_timezone(lat=lat, lon=lon)
        self.tz = get_standard_timezone(tz_name=self.tz_name)

        obs = ephem.Observer()
        obs.lat = str(lat)
        obs.long = str(lon)
        obs.elev = elev

        self.obs = obs

    def get_next_sunrise(self, date, as_local=True):
        """Return the next sunrise at or after `date`."""
        return self._get_rise_set(
            date=date, rise_or_set="rise", next_or_last="next", as_local=as_local
        )

    def get_last_sunrise(self, date, as_local=True):
        """Return the most recent sunrise at or before `date`."""
        return self._get_rise_set(
            date=date, rise_or_set="rise", next_or_last="last", as_local=as_local
        )

    def get_next_sunset(self, date, as_local=True):
        """Return the next sunset at or after `date`."""
        return self._get_rise_set(
            date=date, rise_or_set="set", next_or_last="next", as_local=as_local
        )

    def get_last_sunset(self, date, as_local=True):
        """Return the most recent sunset at or before `date`."""
        return self._get_rise_set(
            date=date, rise_or_set="set", next_or_last="last", as_local=as_local
        )

    def _get_rise_set(self, date, rise_or_set, next_or_last, as_local=True):
        """Look up the requested rise/set event and convert to local time if asked.

        `date` is treated as this site's local (standard, no-DST) time when
        naive — ephem.Observer.date is always UTC, so a naive local value
        must be converted before being handed to ephem, or a query near
        the UTC offset boundary silently resolves against the wrong day
        (e.g. local noon at UTC+9:30 is 21:30 UTC, already past sunset).
        """
        if date.tzinfo is None:
            date = date.replace(tzinfo=self.tz)
        self.obs.date = date.astimezone(UTC)

        func = self.FUNC_MAP[(rise_or_set, next_or_last)]

        dt_utc = func(self.obs, ephem.Sun()).datetime().replace(tzinfo=UTC)

        if as_local:
            return dt_utc.astimezone(self.tz)

        return dt_utc


def get_day_night_binary(lat: float, lon: float, elev: float, dates) -> pd.Series:
    """Return a 1/0 (day/night) series aligned to `dates`, for (lat, lon, elev).

    dates: anything pd.DatetimeIndex(...) accepts — a list of datetime, a
        list/array of numpy.datetime64 (e.g. ds.time.values), an existing
        DatetimeIndex, ISO strings, etc. Normalized once at entry so the
        rest of this function only ever deals with pd.Timestamp.

    Sunrise/sunset is computed once per unique calendar day spanned by
    `dates` (not once per record) and used to mask that day's own slice.
    Both the sunrise and sunset instants themselves are classified as
    night — day is the open interval strictly between them — which is
    immaterial in practice at typical data cadences, since a real
    timestamp essentially never lands exactly on the sunrise/sunset second.
    """
    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0:
        raise ValueError("dates arg must not be empty")

    sun = SunTime(lat, lon, elev)
    bin_series = pd.Series(data=1, index=dates)

    start = datetime.combine(date=dates.min().date(), time=time(12))
    end = datetime.combine(date=dates.max().date(), time=time(12))
    day_dates = pd.date_range(start=start, end=end, freq="D").to_pydatetime()

    for date in day_dates:
        rises = sun.get_last_sunrise(date=date, as_local=True).replace(tzinfo=None)
        sets = sun.get_next_sunset(date=date, as_local=True).replace(tzinfo=None)

        day_start = datetime.combine(date.date(), time.min)
        day_end = day_start + timedelta(days=1)
        in_day = (bin_series.index >= day_start) & (bin_series.index < day_end)

        bin_series.loc[in_day & (bin_series.index <= rises)] = 0
        bin_series.loc[in_day & (bin_series.index >= sets)] = 0

    return bin_series


def get_standard_timezone(tz_name: str) -> timezone:
    """Return a fixed-offset `timezone` for `tz_name`'s standard (non-DST) UTC offset.

    Subtracts the DST component rather than sampling a "winter" date, which
    would be the wrong half of the year in one hemisphere.
    """
    reference = datetime(2026, 1, 1, tzinfo=ZoneInfo(tz_name))
    return timezone(reference.utcoffset() - reference.dst())


def get_timezone(lat: float, lon: float) -> str:
    """Return the IANA timezone name for a (lat, lon) coordinate."""
    return _timezone_finder().timezone_at(lat=lat, lng=lon)
