#!/usr/bin/env python3
"""Sun-position and timezone helpers, built on ephem and timezonefinder."""

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import ephem
from timezonefinder import TimezoneFinder

tzf = TimezoneFinder()


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
        """Look up the requested rise/set event and convert to local time if asked."""
        self.obs.date = date

        func = self.FUNC_MAP[(rise_or_set, next_or_last)]

        dt_utc = func(self.obs, ephem.Sun()).datetime().replace(tzinfo=UTC)

        if as_local:
            return dt_utc.astimezone(self.tz)

        return dt_utc


def get_standard_timezone(tz_name: str) -> timezone:
    """Return a fixed-offset `timezone` for `tz_name`'s winter (non-DST) UTC offset."""
    tz = ZoneInfo(tz_name)

    winter_dt = datetime(2026, 7, 1, tzinfo=tz)

    return timezone(winter_dt.utcoffset())


def get_utc_now(as_iso=False) -> datetime | str:
    """Return the current UTC time, as a datetime or an ISO8601 string."""
    now = datetime.now(UTC)
    if not as_iso:
        return now
    return now.isoformat()


def get_tz_aware_datetime(naive_dt, tz_name, as_iso=False):
    """Attach `tz_name`'s standard-time offset to a naive datetime.

    Args:
        naive_dt: datetime to convert (its own tzinfo, if any, is used as
            the source zone for the conversion — see `datetime.astimezone`).
        tz_name: IANA timezone name (e.g. 'Australia/Darwin').
        as_iso: if True, return an ISO8601 string instead of a datetime.

    Returns:
        Timezone-aware datetime, or its ISO8601 string form if `as_iso`.
    """
    tz_dt = naive_dt.astimezone(get_standard_timezone(tz_name=tz_name))
    if not as_iso:
        return tz_dt
    return tz_dt.isoformat()


def get_local_datetime_now(tz_name, return_tz_aware: bool = True, as_iso: bool = False):
    """Return the current time converted to `tz_name`'s standard-time offset.

    Args:
        tz_name: IANA timezone name (e.g. 'Australia/Darwin').
        return_tz_aware: if False, strip tzinfo from the result.
        as_iso: if True, return an ISO8601 string instead of a datetime.

    Returns:
        Local datetime (aware or naive per `return_tz_aware`), or its
        ISO8601 string form if `as_iso`.
    """
    local = get_tz_aware_datetime(naive_dt=get_utc_now(), tz_name=tz_name)
    if not return_tz_aware:
        local = local.replace(tzinfo=None)
    if not as_iso:
        return local
    return local.isoformat()


def get_timezone(lat: float, lon: float) -> str:
    """Return the IANA timezone name for a (lat, lon) coordinate."""
    return tzf.timezone_at(lat=lat, lng=lon)


def get_UTC_offset(
    tz_name: str,
    date: datetime = None,
    offset_as_delta: bool = False,
    dst: bool = False,
) -> float | timedelta | None:
    """Return `tz_name`'s UTC offset at `date`.

    Args:
        tz_name: IANA timezone name (e.g. 'Australia/Darwin').
        date: datetime to evaluate the offset at (DST varies by date).
            Defaults to now.
        offset_as_delta: if True, return a timedelta instead of hours.
        dst: if True, include any daylight-saving adjustment; if False
            (default), subtract it out to get the standard-time offset.

    Returns:
        UTC offset in hours (or as a timedelta if `offset_as_delta`), or
        None if the timezone has no defined offset at `date`.
    """
    if date is None:
        date = datetime.now()
    tz = ZoneInfo(tz_name)
    offset = tz.utcoffset(date)
    if not dst:
        offset -= tz.dst(date)
    if offset_as_delta:
        return offset
    return offset.total_seconds() / 3600 if offset else None


def format_fast_timestamp(ts: datetime) -> str:
    """Format a sub-second datetime as a Campbell TOA5 timestamp string.

    Rounds to the nearest 100 ms and appends tenths of a second when
    non-zero (e.g. '2025-01-01 00:00:00.1'). Use for high-frequency data
    where DATA_TIME_FORMAT would lose sub-second precision.

    Args:
        ts: datetime, typically a pandas Timestamp from a high-frequency index.

    Returns:
        Formatted string compatible with the Campbell TOA5 timestamp column.
    """
    rounded = ts + timedelta(microseconds=500)
    tenths = rounded.microsecond // 100_000
    if tenths == 0:
        return f"{rounded:%Y-%m-%d %H:%M:%S}"
    return f"{rounded:%Y-%m-%d %H:%M:%S}.{tenths}"
