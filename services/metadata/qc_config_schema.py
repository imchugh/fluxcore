#!/usr/bin/env python3
"""Pydantic schema, loading, and validation for per-site L2 QC YAML configs.

Unlike site_config_schema.py, this module also assembles the frozen contract
object itself (SiteQCConfig) rather than handing off to a separate TERN-adapter
loader: QC config content (canonical variable names + numeric thresholds) needs
no TERN-specific resolution, so the whole load path stays in tier 1 (generic
core) — see CLAUDE.md's "Generic-core / TERN-adapter / ops boundary".

Per-site QC config files live beside the L1 site configs, in the
site_config_files_L2 stream (site_configs/operational/L2/{site_name}.yml), so
site-specific configs stay separate from code. The shared, site-independent
defaults table (configs/qc/_range_defaults.yml) stays in-repo: it changes with
the code and applies to every site.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import xarray as xr
from pydantic import BaseModel, ConfigDict, RootModel, field_validator, model_validator

from domain.enums import StatisticType
from infrastructure import paths
from infrastructure.file_io import read_yml
from services.metadata.core.variable_name_parser import (
    NameParser,
    VariableNameParseError,
)


class RangeCheckSpec(BaseModel):
    """Flag values outside [lower, upper]."""

    model_config = ConfigDict(extra="forbid")

    lower: float
    upper: float

    @model_validator(mode="after")
    def check_order(self):
        """Enforce lower < upper."""
        if self.lower >= self.upper:
            raise ValueError(
                f"range_check lower ({self.lower}) must be < upper ({self.upper})"
            )
        return self


class MADFilterSpec(BaseModel):
    """Median-absolute-deviation despiking, ported from PyFluxPro's do_madfilter."""

    model_config = ConfigDict(extra="forbid")

    reference_var: str
    window_days: int = 13
    fsd_threshold: float = 12.0
    zfc: float = 5.5
    edge_threshold: float | tuple[float, float] = (20.0, 80.0)

    @field_validator("window_days")
    @classmethod
    def check_window_days(cls, v):
        """Enforce a positive window size."""
        if v <= 0:
            raise ValueError("mad_filter.window_days must be > 0")
        return v

    @field_validator("zfc")
    @classmethod
    def check_zfc(cls, v):
        """Enforce a positive z-score scale factor."""
        if v <= 0:
            raise ValueError("mad_filter.zfc must be > 0")
        return v


class VariableQCSpec(BaseModel):
    """The checks configured for one L1 output variable (exact store name)."""

    model_config = ConfigDict(extra="forbid")

    range_check: RangeCheckSpec | None = None
    exclude_dates: list[tuple[datetime, datetime]] | None = None
    dependency_check: list[str] | None = None
    mad_filter: MADFilterSpec | None = None

    @field_validator("exclude_dates")
    @classmethod
    def check_exclude_dates_order(cls, v):
        """Enforce start < end for every exclude_dates range."""
        if v is None:
            return v
        for start, end in v:
            if start > end:
                raise ValueError(
                    f"exclude_dates range start ({start}) must not be after end "
                    f"({end}); use start == end for a single-record exclusion"
                )
        return v

    @field_validator("dependency_check")
    @classmethod
    def check_dependency_check_nonempty(cls, v):
        """Reject an explicitly-empty dependency_check list."""
        if v is not None and len(v) == 0:
            raise ValueError("dependency_check must list at least one variable")
        return v


class QCConfigSchema(RootModel[dict[str, VariableQCSpec]]):
    """Flat per-site QC config: {variable_name: VariableQCSpec}."""


def validate_qc_config_structure(file: Path | str) -> QCConfigSchema:
    """Validate YAML structure and return the schema object."""
    data = read_yml(file_path=file, enforce_unique_keys=True) or {}
    return QCConfigSchema(data)


@dataclass(frozen=True)
class SiteQCConfig:
    """Assembled QC config for one site: the checks to apply, per variable."""

    site_name: str
    variables: dict[str, VariableQCSpec]

    def dependency_graph_order(self) -> list[str]:
        """Topologically sort configured variables by dependency_check edges.

        Dependencies that aren't themselves configured (no VariableQCSpec) are
        not nodes in this graph — qc_pipeline.apply_qc resolves those directly
        from isnull() rather than from an ordered flag. Raises ValueError on a
        cycle among configured variables.
        """
        edges = {
            name: sorted(set(spec.dependency_check or ()) & set(self.variables))
            for name, spec in self.variables.items()
        }
        order: list[str] = []
        state: dict[str, int] = {}  # 0=unvisited (absent), 1=visiting, 2=done

        def visit(name: str, stack: list[str]) -> None:
            if state.get(name) == 2:
                return
            if state.get(name) == 1:
                cycle = " -> ".join([*stack[stack.index(name) :], name])
                raise ValueError(f"Cyclic dependency_check graph detected: {cycle}")
            state[name] = 1
            for dep in edges[name]:
                visit(dep, [*stack, name])
            state[name] = 2
            order.append(name)

        for name in sorted(self.variables):
            visit(name, [])
        return order


def load_qc_config(site_name: str, config_dir: Path | None = None) -> SiteQCConfig:
    """Load a site's QC config, or an empty (pass-through) one if none exists.

    Args:
        site_name: registered site name.
        config_dir: directory containing {site_name}.yml. Defaults to the
            site_config_files_L2 stream (site_configs/operational/L2).
    """
    if config_dir is None:
        config_dir = paths.get_local_stream_path("configs", "site_config_files_L2")
    file_path = Path(config_dir) / f"{site_name}.yml"

    if not file_path.exists():
        return SiteQCConfig(site_name=site_name, variables={})

    schema = validate_qc_config_structure(file_path)
    return SiteQCConfig(site_name=site_name, variables=dict(schema.root))


def load_range_defaults(config_dir: Path | None = None) -> dict:
    """Load the shared range_check defaults table (configs/qc/_range_defaults.yml).

    No structural validation beyond plain YAML parsing -- this file is
    maintained in-repo, not per-site, so it doesn't need the same
    unknown-key/type strictness as a site's own QC config. Each entry is
    either a [min, max] pair, or a dict keyed by qualifier (Diag only) or
    StatisticType suffix (Av/Sd/Vr/...) -- see the file's own header comment
    for the full keying convention. Returns {} if the file doesn't exist.
    """
    config_dir = config_dir or (paths.CONFIG_PATH / "qc")
    file_path = Path(config_dir) / "_range_defaults.yml"

    if not file_path.exists():
        return {}

    return read_yml(file_path=file_path, enforce_unique_keys=True) or {}


def validate_qc_config_variables(
    qc_config: SiteQCConfig, available_variables: Iterable[str]
) -> None:
    """Raise ValueError listing every QC-config variable reference not present.

    Checks top-level variable keys, dependency_check entries, and
    mad_filter.reference_var — anywhere a variable name is referenced. Takes
    available_variables as a plain iterable (no I/O) so the caller can supply
    e.g. an already-open Zarr store's ds.data_vars.
    """
    available = set(available_variables)
    missing = set()

    for name, spec in qc_config.variables.items():
        if name not in available:
            missing.add(name)
        for dep in spec.dependency_check or ():
            if dep not in available:
                missing.add(dep)
        if spec.mad_filter is not None and spec.mad_filter.reference_var not in available:
            missing.add(spec.mad_filter.reference_var)

    if missing:
        raise ValueError(
            f"QC config for site {qc_config.site_name!r} references variable(s) "
            f"not present in the data store: {sorted(missing)}"
        )


def load_dependency_defaults(
    config_dir: Path | None = None,
) -> dict[str, VariableQCSpec]:
    """Load configs/qc/_dependency_defaults.yml.

    Network-wide default dependency_check (and, where needed, range_check)
    patterns keyed by exact L1 output variable name.

    Same shape and validation as a site's own QC config
    (validate_qc_config_structure) -- unlike _range_defaults.yml's looser
    plain-YAML load, a typo here is caught by Pydantic like a bad site file
    would be. Only some sites actually have the variables these patterns
    target (e.g. raw-covariance/Diag_SONIC/Diag_IRGA sites vs EddyPro-only
    sites) -- resolve_qc_config existence-gates each entry against a site's
    real data at merge time. Returns {} if the file doesn't exist.
    """
    config_dir = config_dir or (paths.CONFIG_PATH / "qc")
    file_path = Path(config_dir) / "_dependency_defaults.yml"

    if not file_path.exists():
        return {}

    return dict(validate_qc_config_structure(file_path).root)


def lookup_default_range(
    ds: xr.Dataset,
    var_name: str,
    range_defaults: dict,
    name_parser: NameParser,
) -> tuple[float, float] | None:
    """Resolve var_name's default [min, max] from range_defaults, or None.

    quantity and qualifier come from parsing var_name itself; statistic
    comes from ds[var_name].attrs["statistic_type"] (the authoritative
    source — required for every continuous variable by site_config_schema,
    not something to re-derive from the name, which fails on several real
    canonical names). Returns None (no default, pass through) wherever the
    name doesn't parse, the quantity has no entry, or the entry is a dict
    with no matching qualifier or statistic key.

    Public (used by resolve_qc_config's default-fallback and
    orchestration.legacy_rtmc_export's range-limiting step).
    """
    try:
        parsed = name_parser.parse_variable_name(var_name)
    except VariableNameParseError:
        return None

    entry = range_defaults.get(parsed.quantity)
    if entry is None:
        return None
    if isinstance(entry, list):
        return tuple(entry)

    if parsed.qualifier is not None and parsed.qualifier in entry:
        return tuple(entry[parsed.qualifier])

    statistic_attr = ds[var_name].attrs.get("statistic_type")
    if statistic_attr is not None:
        statistic_suffix = StatisticType(statistic_attr).suffix
        if statistic_suffix in entry:
            return tuple(entry[statistic_suffix])

    return None


def resolve_qc_config(
    qc_config: SiteQCConfig,
    dependency_defaults: dict[str, VariableQCSpec],
    range_defaults: dict,
    ds: xr.Dataset,
) -> SiteQCConfig:
    """Merge network-wide defaults into a site's explicit QC config, field by field.

    For every checkable variable in ds (mirrors qc_pipeline.checkable_variables's
    own _QCFlag/crs exclusion, duplicated inline rather than imported since
    qc_pipeline imports this module and importing back would be circular):
    the site's own explicit VariableQCSpec field wins if set. Otherwise:
      - range_check falls back to dependency_defaults' own range_check (used
        by quality-flag variables like Diag_SONIC/Fco2_QC that have no
        _range_defaults.yml quantity entry of their own), then to
        lookup_default_range.
      - dependency_check falls back to dependency_defaults' entry for that
        exact variable name, existence-gated against ds: an individual
        source name not present in ds is dropped from the list rather than
        dropping the whole entry; the whole entry is dropped only if every
        source turns out unavailable.

    This is deliberately more tolerant than validate_qc_config_variables
    (strict, applied only to qc_config's own raw entries before this merge
    runs): existence-gating here is what exempts e.g. EddyPro-only sites
    from the SONIC/IRGA defaults with no site-name or instrument-type
    branching anywhere in code -- their conditional variables simply aren't
    in ds, so those defaults silently don't attach.

    A variable with no resolved field from any source (site config or
    defaults) is omitted from the result, same as a variable with no config
    entry at all today.
    """
    available = {v for v in ds.data_vars if not v.endswith("_QCFlag") and v != "crs"}
    name_parser = NameParser()
    merged: dict[str, VariableQCSpec] = {}

    for var_name in available:
        site_spec = qc_config.variables.get(var_name)
        default_spec = dependency_defaults.get(var_name)

        range_check = site_spec.range_check if site_spec else None
        if range_check is None and default_spec is not None:
            range_check = default_spec.range_check
        if range_check is None:
            bounds = lookup_default_range(ds, var_name, range_defaults, name_parser)
            if bounds is not None:
                range_check = RangeCheckSpec(lower=bounds[0], upper=bounds[1])

        dependency_check = site_spec.dependency_check if site_spec else None
        if (
            dependency_check is None
            and default_spec is not None
            and default_spec.dependency_check
        ):
            gated = [dep for dep in default_spec.dependency_check if dep in available]
            if gated:
                dependency_check = gated

        exclude_dates = site_spec.exclude_dates if site_spec else None
        mad_filter = site_spec.mad_filter if site_spec else None

        if (
            range_check is None
            and dependency_check is None
            and exclude_dates is None
            and mad_filter is None
        ):
            continue

        merged[var_name] = VariableQCSpec(
            range_check=range_check,
            exclude_dates=exclude_dates,
            dependency_check=dependency_check,
            mad_filter=mad_filter,
        )

    return SiteQCConfig(site_name=qc_config.site_name, variables=merged)
