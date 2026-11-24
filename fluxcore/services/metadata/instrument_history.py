#!/usr/bin/env python3
"""A variable's instrument history: which instrument measured it, and when.

Every variable built from raw inputs carries one, even when its instrument
never changed (a single period), so its `instrument` attr is always derived
from it rather than kept in step separately.

The history has one shape in a dataset's attrs, from assembly through the
Zarr store: the JSON form `to_attr()` writes (lists and fixed-key records
only, because Zarr's attrs writer alphabetises dict keys but keeps list
order). Anything that needs to reason about it parses that form with
`InstrumentHistory.from_attr()`; NetCDF export turns it into the per-year
`instrument` / `instrument_history` strings with `netcdf_attrs()`.

Attr form, roles in config order, periods in time order:

    [{"role": null, "periods": [
        {"instrument": "Rotronic HC2S3", "start": "2025-06-10T09:00:00",
         "end": "2026-01-20T13:30:00"},
        {"instrument": "Vaisala HMP155", "start": "2026-01-20T14:00:00",
         "end": null}]}]

`role` is null for a single-instrument variable, or the compound key
(`sonic_anemometer`, `irga`) for a flux variable; null dates are open-ended.

Public API
----------
InstrumentPeriod, RoleHistory, InstrumentHistory
InstrumentHistory.from_specs(var_specs) / .from_attr(attr) / .to_attr()
InstrumentHistory.current() -> str
InstrumentHistory.netcdf_attrs(window_start, window_end) -> (str, str | None)
"""

from collections.abc import Sequence
from datetime import datetime

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentPeriod:
    """One instrument's period; None dates are open-ended."""

    instrument: str
    start: datetime | None
    end: datetime | None

    def clip(
        self, window_start: datetime, window_end: datetime
    ) -> tuple[datetime, datetime] | None:
        """This period's dates within an inclusive window; None if outside it.

        Open-ended dates resolve to the window's edge: there's no config-only
        answer to when an undated era starts or ends, so the caller's window
        (e.g. a year clamped to the data's actual extent) supplies it.
        """
        start = window_start if self.start is None else max(window_start, self.start)
        end = window_end if self.end is None else min(window_end, self.end)
        if start > window_end or end < window_start:
            return None
        return start, end


@dataclass(frozen=True)
class RoleHistory:
    """The periods (in time order) of one instrument role."""

    role: str | None
    periods: tuple[InstrumentPeriod, ...]

    @property
    def changed(self) -> bool:
        """Whether this role's instrument changed over the full history."""
        return len(self.periods) > 1


@dataclass(frozen=True)
class InstrumentHistory:
    """A variable's instrument history: one RoleHistory per instrument role."""

    roles: tuple[RoleHistory, ...]

    # Construction

    @classmethod
    def from_specs(cls, var_specs: Sequence) -> "InstrumentHistory":
        """Build from a variable's VariableSpecs, in time (config) order.

        Consecutive inputs from one instrument (e.g. a units change on the
        same sensor) are one period, from the first's begin to the last's
        end. An instrument can't return after a changeover (rejected by
        runtime_config_builder), so each role's instruments are distinct.
        """
        first = var_specs[0].instrument
        roles = list(first) if isinstance(first, dict) else [None]
        for spec in var_specs:
            if isinstance(spec.instrument, dict) != isinstance(first, dict):
                raise ValueError(
                    f"{spec.canonical_name}: inputs mix single and compound instruments"
                )
        return cls(
            roles=tuple(
                RoleHistory(
                    role=role,
                    periods=_merge_consecutive(
                        [
                            (_role_instrument(spec, role), spec.begin, spec.end)
                            for spec in var_specs
                        ]
                    ),
                )
                for role in roles
            )
        )

    @classmethod
    def from_attr(cls, attr: list) -> "InstrumentHistory":
        """Parse the attr form written by `to_attr()`."""
        if not (isinstance(attr, list) and attr and all(_is_role(r) for r in attr)):
            raise ValueError(
                f"Unrecognised instrument_history attr {attr!r}: written by an "
                "older fluxcore? Rebuild the store."
            )
        return cls(
            roles=tuple(
                RoleHistory(
                    role=r["role"],
                    periods=tuple(
                        InstrumentPeriod(
                            instrument=p["instrument"],
                            start=_parse(p["start"]),
                            end=_parse(p["end"]),
                        )
                        for p in r["periods"]
                    ),
                )
                for r in attr
            )
        )

    # Output

    def to_attr(self) -> list:
        """JSON-safe, order-preserving attr form (see module docstring)."""
        return [
            {
                "role": r.role,
                "periods": [
                    {
                        "instrument": p.instrument,
                        "start": _isoformat(p.start),
                        "end": _isoformat(p.end),
                    }
                    for p in r.periods
                ],
            }
            for r in self.roles
        ]

    def current(self) -> str:
        """Latest instrument per role, comma-joined in role order."""
        return ",".join(r.periods[-1].instrument for r in self.roles)

    def netcdf_attrs(
        self, window_start: datetime, window_end: datetime
    ) -> tuple[str, str | None]:
        """NetCDF `instrument` and `instrument_history` strings for a window.

        instrument: per role, the last instrument in use within the window
        (or overall, if none of its periods falls in the window),
        comma-joined in role order.

        instrument_history (None means omit the attr):
          - single instrument: '(inst,start,end)|(inst,start,end)', only
            when two or more periods fall in the window;
          - compound: 'role>(inst,start,end)|...;role>...', one segment per
            role whose instrument ever changed and has a period in the
            window.
        """
        instruments = []
        segments = []
        for r in self.roles:
            clipped = [
                (p.instrument, dates)
                for p in r.periods
                if (dates := p.clip(window_start, window_end)) is not None
            ]
            instruments.append(clipped[-1][0] if clipped else r.periods[-1].instrument)
            formatted = "|".join(
                f"({inst},{start.isoformat()},{end.isoformat()})"
                for inst, (start, end) in clipped
            )
            if r.role is None:
                if len(clipped) > 1:
                    segments.append(formatted)
            elif r.changed and clipped:
                segments.append(f"{r.role}>{formatted}")

        return ",".join(instruments), ";".join(segments) or None


def _merge_consecutive(
    periods: list[tuple[str, datetime | str | None, datetime | str | None]],
) -> tuple[InstrumentPeriod, ...]:
    """Collapse runs of one instrument into one period each."""
    merged: list[InstrumentPeriod] = []
    for instrument, begin, end in periods:
        begin, end = _parse(begin), _parse(end)
        if merged and merged[-1].instrument == instrument:
            merged[-1] = InstrumentPeriod(
                instrument=instrument, start=merged[-1].start, end=end
            )
        else:
            merged.append(InstrumentPeriod(instrument=instrument, start=begin, end=end))
    return tuple(merged)


def _role_instrument(spec, role: str | None) -> str:
    """The spec's instrument for a role (None: a single-instrument spec)."""
    return spec.instrument if role is None else spec.instrument[role]


def _is_role(record) -> bool:
    return isinstance(record, dict) and {"role", "periods"} <= record.keys()


def _parse(value: datetime | str | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _isoformat(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()
