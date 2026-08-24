#!/usr/bin/env python3
"""Pydantic schema, loading, and validation for per-site L2 QC YAML configs.

Unlike site_config_schema.py, this module also assembles the frozen contract
object itself (SiteQCConfig) rather than handing off to a separate TERN-adapter
loader: QC config content (canonical variable names + numeric thresholds) needs
no TERN-specific resolution, so the whole load path stays in tier 1 (generic
core) — see CLAUDE.md's "Generic-core / TERN-adapter / ops boundary".

QC config files live in-repo at configs/qc/{site_name}.yml, alongside the other
tier-1 config content (canonical_quantities.yml etc.) — not in TERN's externally
managed site_configs/ tree, since QC thresholds are pure generic-core content.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, RootModel, field_validator, model_validator

from infrastructure import paths
from infrastructure.file_io import read_yml


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
        config_dir: directory containing {site_name}.yml. Defaults to
            infrastructure.paths.CONFIG_PATH / "qc".
    """
    config_dir = config_dir or (paths.CONFIG_PATH / "qc")
    file_path = Path(config_dir) / f"{site_name}.yml"

    if not file_path.exists():
        return SiteQCConfig(site_name=site_name, variables={})

    schema = validate_qc_config_structure(file_path)
    return SiteQCConfig(site_name=site_name, variables=dict(schema.root))


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
