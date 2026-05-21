#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 08:41:42 2026

@author: imchugh
"""

# -----------------------------------------------------------------------------
# IMPORTS 
# -----------------------------------------------------------------------------

import ephem
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder

# -----------------------------------------------------------------------------
# INITS
# -----------------------------------------------------------------------------

tzf = TimezoneFinder()

# -----------------------------------------------------------------------------
# CLASSES
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

class SunTime:
    """
    Simple class to get sunrise and sunset times.
    """

    FUNC_MAP = {
        ('rise', 'next'): ephem.Observer.next_rising,
        ('rise', 'last'): ephem.Observer.previous_rising,
        ('set', 'next'): ephem.Observer.next_setting,
        ('set', 'last'): ephem.Observer.previous_setting,
        }

    #--------------------------------------------------------------------------
    def __init__(self, lat: float, lon: float, elev: float) -> None:
        """
        Initialise.
        """

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

    #--------------------------------------------------------------------------
    def get_next_sunrise(self, date, as_local=True):

        return self._get_rise_set(
            date=date,
            rise_or_set='rise',
            next_or_last='next',
            as_local=as_local
            )

    #--------------------------------------------------------------------------
    def get_last_sunrise(self, date, as_local=True):

        return self._get_rise_set(
            date=date,
            rise_or_set='rise',
            next_or_last='last',
            as_local=as_local
            )

    #--------------------------------------------------------------------------
    def get_next_sunset(self, date, as_local=True):

        return self._get_rise_set(
            date=date,
            rise_or_set='set',
            next_or_last='next',
            as_local=as_local
            )

    #--------------------------------------------------------------------------
    def get_last_sunset(self, date, as_local=True):

        return self._get_rise_set(
            date=date,
            rise_or_set='set',
            next_or_last='last',
            as_local=as_local
            )

    #--------------------------------------------------------------------------
    def _get_rise_set(
        self,
        date,
        rise_or_set,
        next_or_last,
        as_local=True
        ):

        self.obs.date = date

        func = self.FUNC_MAP[(rise_or_set, next_or_last)]

        dt_utc = (
            func(self.obs, ephem.Sun())
            .datetime()
            .replace(tzinfo=timezone.utc)
            )

        if as_local:
            return dt_utc.astimezone(self.tz)

        return dt_utc
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def get_standard_timezone(tz_name: str):

    tz = ZoneInfo(tz_name)

    winter_dt = datetime(2026, 7, 1, tzinfo=tz)

    return timezone(winter_dt.utcoffset())
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def get_utc_now(as_iso=False) -> str:
    """
    Generate UTC ISO8601 timestamp.
    """

    now = datetime.now(timezone.utc)
    if not as_iso:
        return now
    return now.isoformat()
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def get_tz_aware_datetime(naive_dt, tz_name, as_iso=False):
    
    tz_dt = naive_dt.astimezone(get_standard_timezone(tz_name=tz_name))
    if not as_iso:
        return tz_dt
    return tz_dt.isoformat()
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def get_local_datetime_now(
        tz_name, return_tz_aware: bool=True, as_iso: bool=False
        ):
    
    local = get_tz_aware_datetime(
        naive_dt=get_utc_now(), 
        tz_name=tz_name
        )
    if not return_tz_aware:
        local = local.replace(tzinfo=None)
    if not as_iso:
        return local
    return local.isoformat()        
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def get_timezone(lat: float, lon: float) -> str:
    """
    Get the name of the timezone based on coordinates.
    """
    
    return tzf.timezone_at(lat=lat, lng=lon)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def get_UTC_offset(
    tz_name: str, 
    date: datetime=None, 
    offset_as_delta: bool=False, 
    dst: bool=False
    ) -> float | timedelta | None:
    
    if date is None:
        date = datetime.now()
    tz = ZoneInfo(tz_name)
    offset = tz.utcoffset(date)
    if not dst:
        offset -= tz.dst(date)
    if offset_as_delta:
        return offset
    return offset.total_seconds() / 3600 if offset else None
# -----------------------------------------------------------------------------
