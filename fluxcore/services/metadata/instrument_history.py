#!/usr/bin/env python3
"""A variable's instrument history: which instrument measured it, and when.

It is also the record of which periods were calculated rather than measured
(`calculated_from`, see below), so it says where every value came from.

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

A period whose values were calculated rather than measured also has
`calculated_from`: the canonical variables they were calculated from. Either
a site config's calculated input (`calculate_from:`, e.g. an old logger that
output only the AH it calculated from its HMP's RH, so RH's old era is
calculated from AH and Ta), or fluxcore's own whole-record derivations
(derived_quantities: CO2 mole fraction, the missing one of AH/RH, Fn), which
mark every period of the history they copy from their source:

        {"instrument": "Vaisala HMP45C", "start": null,
         "end": "2025-09-08T17:30:00",
         "calculated_from": ["AH_23m", "Ta_23m"]}

The key is absent from a measured period. A change between measured and
calculated starts a new period even on the same instrument. In NetCDF a
calculated period reads '(Vaisala HMP45C calculated from AH_23m Ta_23m,
start,end)', and a year with any calculated period always gets an
`instrument_history` attr, even with one instrument.

Instrument URIs live beside the history, in the variable's `instrument_uri`
attr: one `{instrument name: uri | null}` map over every instrument in the
history (`instrument_uri_map()`). A URI is resolved from the instrument
name, so the name is the only key it needs; the history says which name
applies when. NetCDF export looks each window's instruments up in it
(`InstrumentHistory.netcdf_uri()`).

Public API
----------
InstrumentPeriod, RoleHistory, InstrumentHistory
InstrumentHistory.from_specs(var_specs) / .from_attr(attr) / .to_attr()
InstrumentHistory.calculated(calculated_from) -> InstrumentHistory
InstrumentHistory.current() -> str
InstrumentHistory.netcdf_attrs(window_start, window_end) -> (str, str | None)
InstrumentHistory.netcdf_uri(uris, window_start, window_end) -> str | None
instrument_uri_map(var_specs) -> dict[str, str | None]
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
    calculated_from: tuple[str, ...] | None = None

    @property
    def label(self) -> str:
        """The period's NetCDF label: the instrument, plus any calculation."""
        if self.calculated_from is None:
            return self.instrument
        return f"{self.instrument} calculated from {' '.join(self.calculated_from)}"

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
        end, unless one is measured and the other calculated (a spec's
        calculate_from). An instrument can't return after a changeover
        (rejected by runtime_config_builder), so each role's instruments are
        distinct.
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
                            (
                                _role_instrument(spec, role),
                                spec.begin,
                                spec.end,
                                spec.calculate_from,
                            )
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
                            calculated_from=_sources(p.get("calculated_from")),
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
                "periods": [_period_attr(p) for p in r.periods],
            }
            for r in self.roles
        ]

    def calculated(self, calculated_from: Sequence[str]) -> "InstrumentHistory":
        """This history with every period calculated from `calculated_from`.

        For a variable derived over its whole record (derived_quantities):
        it keeps its source's instruments and dates, all calculated.
        """
        sources = tuple(calculated_from)
        return InstrumentHistory(
            roles=tuple(
                RoleHistory(
                    role=r.role,
                    periods=tuple(
                        InstrumentPeriod(
                            instrument=p.instrument,
                            start=p.start,
                            end=p.end,
                            calculated_from=sources,
                        )
                        for p in r.periods
                    ),
                )
                for r in self.roles
            )
        )

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
            when two or more periods fall in the window, or any of them is
            calculated;
          - compound: 'role>(inst,start,end)|...;role>...', one segment per
            role whose instrument ever changed, or with a calculated period
            in the window, that has a period in the window.
        A calculated period's inst reads 'inst calculated from A B'.
        """
        segments = []
        for r in self.roles:
            clipped = _clipped(r, window_start, window_end)
            formatted = "|".join(
                f"({p.label},{start.isoformat()},{end.isoformat()})"
                for p, (start, end) in clipped
            )
            calculated = any(p.calculated_from for p, _ in clipped)
            if r.role is None:
                if len(clipped) > 1 or calculated:
                    segments.append(formatted)
            elif (r.changed or calculated) and clipped:
                segments.append(f"{r.role}>{formatted}")

        instrument = ",".join(self._window_instruments(window_start, window_end))
        return instrument, ";".join(segments) or None

    def netcdf_uri(
        self,
        uris: dict[str, str | None] | str | None,
        window_start: datetime,
        window_end: datetime,
    ) -> str | None:
        """NetCDF `instrument_uri` for a window (None means omit the attr).

        The URIs of the window's instruments (see netcdf_attrs), comma-joined
        in role order. An unresolved URI keeps its position as an empty item,
        so the string lines up with `instrument` item by item; None if no
        instrument resolves at all. A plain string `uris` (a store written
        before the map form) is returned as it is.
        """
        if uris is None or isinstance(uris, str):
            return uris
        items = [
            uris.get(name) or ""
            for name in self._window_instruments(window_start, window_end)
        ]
        return ",".join(items) if any(items) else None

    def _window_instruments(
        self, window_start: datetime, window_end: datetime
    ) -> list[str]:
        """Per role, the last instrument in use within the window (or overall)."""
        instruments = []
        for r in self.roles:
            clipped = _clipped(r, window_start, window_end)
            instruments.append(
                clipped[-1][0].instrument if clipped else r.periods[-1].instrument
            )
        return instruments


def instrument_uri_map(var_specs: Sequence) -> dict[str, str | None]:
    """`{instrument name: uri | None}` over every instrument in var_specs.

    Specs carry a URI string for a single instrument, or a role-keyed dict
    for a compound one; either way each instrument name maps to one URI
    (URIs are resolved from names), so roles aren't needed in the map.
    """
    uris: dict[str, str | None] = {}
    for spec in var_specs:
        if isinstance(spec.instrument, dict):
            role_uris = spec.instrument_uri or {}
            pairs = [
                (name, role_uris.get(role)) for role, name in spec.instrument.items()
            ]
        else:
            pairs = [(spec.instrument, spec.instrument_uri)]
        for name, uri in pairs:
            if uris.get(name) is None:
                uris[name] = uri
    return uris


def _clipped(
    role: RoleHistory, window_start: datetime, window_end: datetime
) -> list[tuple[InstrumentPeriod, tuple[datetime, datetime]]]:
    """A role's (period, clipped dates) for its periods inside the window."""
    return [
        (p, dates)
        for p in role.periods
        if (dates := p.clip(window_start, window_end)) is not None
    ]


def _merge_consecutive(periods: list[tuple]) -> tuple[InstrumentPeriod, ...]:
    """Collapse runs of one instrument (and one derivation) into one period each.

    `periods` is (instrument, begin, end, calculate_from) per input, in
    time order.
    """
    merged: list[InstrumentPeriod] = []
    for instrument, begin, end, calculated_from in periods:
        begin, end = _parse(begin), _parse(end)
        sources = _sources(calculated_from)
        if (
            merged
            and merged[-1].instrument == instrument
            and merged[-1].calculated_from == sources
        ):
            merged[-1] = InstrumentPeriod(
                instrument=instrument,
                start=merged[-1].start,
                end=end,
                calculated_from=sources,
            )
        else:
            merged.append(
                InstrumentPeriod(
                    instrument=instrument,
                    start=begin,
                    end=end,
                    calculated_from=sources,
                )
            )
    return tuple(merged)


def _period_attr(p: InstrumentPeriod) -> dict:
    """One period's attr record; calculated_from only on a calculated period."""
    record = {
        "instrument": p.instrument,
        "start": _isoformat(p.start),
        "end": _isoformat(p.end),
    }
    if p.calculated_from is not None:
        record["calculated_from"] = list(p.calculated_from)
    return record


def _sources(value) -> tuple[str, ...] | None:
    return None if value is None else tuple(value)


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
