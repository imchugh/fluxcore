#!/usr/bin/env python3
"""Pydantic schema models and structural validation for site YAML configs."""

import re
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from fluxcore.domain.enums import (
    DiagnosticType,
    FluxSystemType,
    LoggerType,
    StatisticType,
    VariableType,
)
from fluxcore.infrastructure.file_io import read_yml

# Input-level configuration


class InputVariableConfig(BaseModel):
    """Schema and validation rules for one raw input variable.

    One source of a variable: a column (name) in a file over a period. A
    variable's inputs are a list, so the same column can feed it from two
    files (a logger change that kept the column name), or from one file
    over consecutive periods with different units or instrument.

    A calculated input (`calculate_from`) is instead a period in which the
    variable is calculated from other configured variables, for an era whose
    logger never output it (e.g. an old program that logged only the AH it
    calculated from its HMP's RH). It names no column, file or units, but
    still names the instrument the values originate from (the source
    variables' sensor). The calculation is the registered one for the
    variable's quantity; the output marks the period calculated in the
    variable's instrument_history (`calculated_from`).
    """

    model_config = ConfigDict(extra="forbid")

    # Define attributes

    # Mandatory
    instrument: str | dict[str, str]

    # Mandatory for a measured input; absent for a calculated one
    name: str | None = None
    file: str | None = None
    units: str | None = None

    # Calculated input: the canonical variables this period is calculated from
    calculate_from: list[str] | None = None

    # Optional
    diag_type: DiagnosticType | None = None
    # The form this input's raw data is stored in, when it differs from the
    # variable's statistic_type: lets a variance and a standard deviation of
    # the same quantity be stitched across a logger change. Defaults to the
    # variable's statistic_type; only variance <-> standard_deviation may differ.
    statistic_type: StatisticType | None = None
    begin: datetime | str | None = None
    end: datetime | str | None = None

    @field_validator("begin", "end", mode="before")
    def parse_datetime(cls, v):
        """Ensure that date fields are either datetimes or None."""
        if v is None:
            return v
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            if v.lower() in {"none", "null", ""}:
                return None
            return datetime.fromisoformat(v)
        raise TypeError(f"Invalid type for datetime field: {type(v)}")

    @property
    def calculated(self) -> bool:
        """Whether this input is a calculated period rather than a raw column."""
        return self.calculate_from is not None

    @property
    def label(self) -> str:
        """The input's name for messages: its column, or its calculation."""
        if self.calculated:
            return f"calculated from {', '.join(self.calculate_from)}"
        return self.name

    @model_validator(mode="after")
    def validate_measured_or_calculated(self):
        """Require a raw column (name/file/units) or calculate_from, not both."""
        raw_fields = {"name": self.name, "file": self.file, "units": self.units}
        if self.calculated:
            if not self.calculate_from:
                raise ValueError("calculate_from must name at least one variable")
            if len(set(self.calculate_from)) != len(self.calculate_from):
                raise ValueError(
                    f"calculate_from lists a variable twice: {self.calculate_from}"
                )
            given = [k for k, v in raw_fields.items() if v is not None]
            for key in ("diag_type", "statistic_type"):
                if getattr(self, key) is not None:
                    given.append(key)
            if given:
                raise ValueError(
                    "A calculated input (calculate_from) has no raw column, so "
                    f"it cannot define {', '.join(given)}"
                )
        else:
            missing = [k for k, v in raw_fields.items() if v is None]
            if missing:
                raise ValueError(
                    f"Input must define {', '.join(missing)} "
                    "(or calculate_from, for a calculated period)"
                )
        return self


class HorizontalLocationConfig(BaseModel):
    """Schema and validation rules for horizontal instrument location."""

    model_config = ConfigDict(extra="forbid")

    # Define attributes

    # All optional
    belowground: dict[str, str] | None = None
    aboveground: dict[str, str] | None = None

    @field_validator("belowground", "aboveground")
    def validate_keys(cls, v):
        """Ensure that horizontal location keys are alphabetic."""
        if v is None:
            return v

        for key in v:
            if not re.fullmatch(r"[a-zA-Z]", key):
                raise ValueError(
                    "Horizontal location keys must be single alphabetic "
                    f"characters, got '{key}'"
                )
        return v

    @model_validator(mode="after")
    def ensure_at_least_one_category(self):
        """Ensure that at least one expected field is in structure."""
        if not self.belowground and not self.aboveground:
            raise ValueError(
                "horizontal_location must contain at least one of "
                "'belowground' or 'aboveground'"
            )
        return self


class CustomMetadataConfig(BaseModel):
    """Schema for custom (site-specific) metadata."""

    # Deliberately open: this is the one model meant to hold site-specific
    # fields the schema doesn't anticipate. Do not tighten to "forbid".
    model_config = ConfigDict(extra="allow")

    # Define attributes

    # Mandatory
    horizontal_location: HorizontalLocationConfig


# Output-level variable configuration


class VariableConfig(BaseModel):
    """Schema for one canonical variable's full configuration."""

    model_config = ConfigDict(extra="forbid")

    # Define attributes

    # Mandatory
    height: float
    input_variables: list[InputVariableConfig] = Field(min_length=1)

    # Optional in yml, explicit in config
    statistic_type: StatisticType | None = None
    variable_type: VariableType = VariableType.CONTINUOUS

    # Optional outright
    height_range: tuple[float, float] | None = None
    standard_name: str | None = None

    @field_validator("height_range")
    @classmethod
    def validate_height_range(cls, v):
        """Enforce ordering by ascending absolute value.

        Depths are negative, so e.g. [0.0, -0.3] is valid (0.0 < 0.3 in
        absolute terms) but [-0.3, 0.0] or [0.0, 0.3] with equal magnitudes
        are not.  Exactly-two-element constraint is enforced by the type
        annotation tuple[float, float].
        """
        if v is None:
            return v
        if abs(v[0]) >= abs(v[1]):
            raise ValueError(
                "height_range must be ordered by ascending absolute value "
                f"(|first| < |second|); got [{v[0]}, {v[1]}]"
            )
        return v

    @model_validator(mode="after")
    def validate_statistic_requirement(self):
        """Require statistic_type for continuous variables; forbid it otherwise."""
        if self.variable_type == VariableType.CONTINUOUS:
            if self.statistic_type is None:
                raise ValueError("Continuous variables must define statistic_type")

        if self.variable_type in (
            VariableType.QUALITY_FLAG,
            VariableType.COUNTER,
        ):
            if self.statistic_type is not None:
                raise ValueError(
                    f"{self.variable_type.value} variables cannot define statistic_type"
                )

        return self

    @model_validator(mode="after")
    def validate_input_statistic_types(self):
        """Allow an input's raw statistic to differ only as variance <-> stdev.

        Taking a square root is the only statistic conversion the pipeline
        can do, so any other mismatch (e.g. a sum input on an average
        variable) is a config error.
        """
        interchangeable = {StatisticType.VAR, StatisticType.STDEV}
        for input_cfg in self.input_variables:
            input_name = input_cfg.label
            input_stat = input_cfg.statistic_type
            if input_stat is None or input_stat == self.statistic_type:
                continue
            if self.variable_type != VariableType.CONTINUOUS:
                raise ValueError(
                    f"Input '{input_name}': {self.variable_type.value} "
                    "variables cannot define an input statistic_type"
                )
            if {input_stat, self.statistic_type} != interchangeable:
                raise ValueError(
                    f"Input '{input_name}': statistic_type '{input_stat.value}' "
                    f"cannot feed a '{self.statistic_type.value}' variable; "
                    "only variance and standard_deviation are interchangeable"
                )
        return self

    @model_validator(mode="after")
    def validate_same_source_periods_disjoint(self):
        """Reject inputs reading one column from one file over overlapping periods.

        The same column may feed a variable more than once, from different
        files (a logger change that kept the column name) or from one file
        over consecutive periods (the column's units or instrument changed),
        but two inputs can't both supply the same timestamps. A missing
        begin/end is open-ended; begin and end are both inclusive.
        """
        by_source: dict[tuple[str, str], list[InputVariableConfig]] = {}
        for input_cfg in self.input_variables:
            if input_cfg.calculated:
                continue
            by_source.setdefault((input_cfg.name, input_cfg.file), []).append(input_cfg)
        for (name, file), inputs in by_source.items():
            for i, first in enumerate(inputs):
                for second in inputs[i + 1 :]:
                    if _periods_overlap(first, second):
                        raise ValueError(
                            f"Input '{name}' from file '{file}' is listed for "
                            f"overlapping periods ({_period(first)} and "
                            f"{_period(second)})"
                        )
        return self

    @model_validator(mode="after")
    def validate_calculated_periods_disjoint(self):
        """Reject a calculated input overlapping any other input of the variable.

        A calculated period fills an era with no raw column, so it must not
        compete with a measured input (or another calculation) for the same
        timestamps.
        """
        for i, first in enumerate(self.input_variables):
            for second in self.input_variables[i + 1 :]:
                if not (first.calculated or second.calculated):
                    continue
                if _periods_overlap(first, second):
                    raise ValueError(
                        f"Input '{first.label}' ({_period(first)}) overlaps "
                        f"input '{second.label}' ({_period(second)}); a "
                        "calculated period must not overlap another input"
                    )
        return self


def _periods_overlap(a: InputVariableConfig, b: InputVariableConfig) -> bool:
    """Whether two inputs' inclusive begin/end periods share any time."""
    a_starts_by_b_end = a.begin is None or b.end is None or a.begin <= b.end
    b_starts_by_a_end = b.begin is None or a.end is None or b.begin <= a.end
    return a_starts_by_b_end and b_starts_by_a_end


def _period(input_cfg: InputVariableConfig) -> str:
    """An input's period for error messages, e.g. 'start to 2020-01-01 00:00'."""

    def fmt(value, open_end):
        return open_end if value is None else value.strftime("%Y-%m-%d %H:%M")

    return f"{fmt(input_cfg.begin, 'start')} to {fmt(input_cfg.end, 'end')}"


# Standalone (file-in) sections: used by run_L1, never set by a network adapter


class SiteMetadataSchema(BaseModel):
    """Schema for one site's metadata: a standalone YAML file or a control file section.

    Required fields are the minimum a site needs to actually be processed:
    location (for CRS/NetCDF export) and time_step/freq_hz (for n_samples,
    dataframe assembly). Everything else is descriptive and commonly sparse
    in real records (e.g. a real site's record may have no
    date_decommissioned/time_zone/UTC_offset).
    """

    site_name: str
    latitude: float
    longitude: float
    time_step: int
    freq_hz: int

    id: str | None = None
    fluxnet_id: str | None = None
    date_commissioned: datetime | None = None
    date_decommissioned: datetime | None = None
    elevation: float | None = None
    soil: str | None = None
    tower_height: float | None = None
    vegetation: str | None = None
    canopy_height: float | None = None
    time_zone: str | None = None
    UTC_offset: float | None = None

    @field_validator("time_step", "freq_hz", mode="before")
    @classmethod
    def coerce_float_string(cls, v):
        """Accept float-string values ('30.0') as real-world records do."""
        if isinstance(v, str):
            return int(float(v))
        return v


class L1PathsConfig(BaseModel):
    """Directories for a standalone L1 run (see run_L1).

    Keys match run_L1's arguments, and an argument wins over its key.
    Relative paths resolve against the control file's own directory.
    """

    model_config = ConfigDict(extra="forbid")

    input_data_path: str | None = None
    input_patch_path: str | None = None
    output_dir: str | None = None


# Complete site configuration
class SiteConfig(BaseModel):
    """Schema and cross-field validation rules for a complete site config.

    The same file is the L1 control file for a standalone run (run_L1),
    which may add the optional `paths`, `site_metadata` and `global_attrs`
    sections. A network adapter supplies those itself and never sets them.
    """

    model_config = ConfigDict(extra="forbid")

    # Define attributes

    site: str
    # The YAML key is file_formats, but its values name the logger family
    # (CSI, LICOR), not a file format: see enums.DEFAULT_FILE_FORMAT.
    logger_types: dict[str, LoggerType] = Field(alias="file_formats")
    flux_system: FluxSystemType
    flux_file: str
    variables: dict[str, VariableConfig]

    custom_metadata: CustomMetadataConfig | None = None

    # Standalone (file-in) sections; see run_L1
    paths: L1PathsConfig | None = None
    site_metadata: SiteMetadataSchema | None = None
    global_attrs: dict[str, str | int | float] | None = None

    # class-level constants for validation
    diag_prefixes: ClassVar[list[str]] = ["Diag_"]

    @model_validator(mode="before")
    @classmethod
    def default_metadata_site_name(cls, data):
        """Default site_metadata's site_name to `site`, and reject a mismatch."""
        if not isinstance(data, dict):
            return data
        metadata = data.get("site_metadata")
        if not isinstance(metadata, dict):
            return data
        site = data.get("site")
        name = metadata.get("site_name")
        if name is None:
            data = {**data, "site_metadata": {**metadata, "site_name": site}}
        elif name != site:
            raise ValueError(
                f"site_metadata.site_name '{name}' does not match site '{site}'"
            )
        return data

    @model_validator(mode="after")
    def enforce_diag_rules(self):
        """Enforce that counter variables declare a diag_type."""
        for var_name, var_cfg in self.variables.items():
            if var_cfg.variable_type == VariableType.COUNTER:
                for input_cfg in var_cfg.input_variables:
                    if input_cfg.diag_type is None:
                        raise ValueError(
                            f"Counter variable '{var_name}' input "
                            f"'{input_cfg.label}' must specify diag_type"
                        )
        return self

    @model_validator(mode="after")
    def enforce_diag_prefix_rules(self):
        """Enforce that Diag_-prefixed variables are declared as counter type."""
        for var_name, var_cfg in self.variables.items():
            if any(var_name.startswith(p) for p in self.diag_prefixes):
                if var_cfg.variable_type != VariableType.COUNTER:
                    raise ValueError(
                        f"Variable '{var_name}' has a diagnostic prefix but "
                        f"variable_type is '{var_cfg.variable_type.value}'; "
                        "expected 'counter'"
                    )
        return self

    @model_validator(mode="after")
    def enforce_name_type_consistency(self):
        """Enforce bidirectional agreement between variable name suffix and type.

        Rules:
          - If the name ends with a known type suffix (e.g. _Ct, _QC), the
            declared variable_type must match.
          - If the declared variable_type carries a suffix, the name must end
            with that suffix — except for Diag_-prefixed variables, which are
            counter by convention without the _Ct suffix.
        """
        suffix_to_type = {vt.suffix: vt for vt in VariableType if vt.suffix is not None}

        for var_name, var_cfg in self.variables.items():
            vtype = var_cfg.variable_type
            is_diag = any(var_name.startswith(p) for p in self.diag_prefixes)

            # Reverse check: name carries a known suffix → type must match
            for suffix, expected_type in suffix_to_type.items():
                if var_name.endswith(f"_{suffix}"):
                    if vtype != expected_type:
                        raise ValueError(
                            f"Variable '{var_name}' name implies type "
                            f"'{expected_type.value}' (suffix '_{suffix}') "
                            f"but variable_type is '{vtype.value}'"
                        )
                    break

            # Forward check: type carries a suffix → name must end with it
            # Diag_-prefixed variables are the conventional exception
            if vtype.suffix is not None and not is_diag:
                if not var_name.endswith(f"_{vtype.suffix}"):
                    raise ValueError(
                        f"Variable '{var_name}' has variable_type "
                        f"'{vtype.value}' but name does not end with "
                        f"'_{vtype.suffix}'"
                    )

        return self

    @model_validator(mode="after")
    def enforce_flux_file_consistency(self):
        """Require flux_file to be mapped to at least one variable's input file."""
        file_list = set()

        for var_cfg in self.variables.values():
            for input_cfg in var_cfg.input_variables:
                if not input_cfg.calculated:
                    file_list.add(input_cfg.file)

        if self.flux_file not in file_list:
            raise ValueError("Flux file must be mapped to at least one variable!")

        return self

    @model_validator(mode="after")
    def enforce_file_formats_complete(self):
        """Require file_formats to list exactly the files the variables use."""
        used = {
            input_cfg.file
            for var_cfg in self.variables.values()
            for input_cfg in var_cfg.input_variables
            if not input_cfg.calculated
        }
        declared = set(self.logger_types)
        if used - declared:
            raise ValueError(
                "Input variables use files not declared in file_formats: "
                f"{sorted(used - declared)}"
            )
        if declared - used:
            raise ValueError(
                "file_formats declares files no input variable uses: "
                f"{sorted(declared - used)}"
            )
        return self

    @model_validator(mode="after")
    def enforce_calculated_sources(self):
        """A calculated input's sources must be other, measured, variables.

        Each source must be a configured variable other than the one being
        calculated, and must not itself have a calculated input: sources are
        read from the assembled measured data, so calculations don't chain.
        Whether the sources' quantities fit the calculation is checked by
        runtime_config_builder, which knows the quantities.
        """
        calculated_vars = {
            name
            for name, var_cfg in self.variables.items()
            if any(i.calculated for i in var_cfg.input_variables)
        }
        for var_name, var_cfg in self.variables.items():
            for input_cfg in var_cfg.input_variables:
                if not input_cfg.calculated:
                    continue
                for source in input_cfg.calculate_from:
                    if source == var_name:
                        raise ValueError(
                            f"Variable '{var_name}' cannot be calculated from itself"
                        )
                    if source not in self.variables:
                        raise ValueError(
                            f"Variable '{var_name}': calculate_from source "
                            f"'{source}' is not a configured variable"
                        )
                    if source in calculated_vars:
                        raise ValueError(
                            f"Variable '{var_name}': calculate_from source "
                            f"'{source}' is itself (partly) calculated; "
                            "calculations can't chain"
                        )
        return self

    # Canonical EC flux quantities: the sole source for the site's flux-system
    # sonic/irga instrument identity (see enforce_compound_instrument_consistency).
    flux_quantities: ClassVar[set[str]] = {"Fco2", "Fh", "Fe"}

    @model_validator(mode="after")
    def enforce_compound_instrument_consistency(self):
        """Ensure Fco2/Fh/Fe agree on the site's flux-system instrument(s).

        Only `Fco2`/`Fh`/`Fe` are inspected — not every dict-shaped
        `instrument` entry in the config — because the same
        `sonic_anemometer`/other-key combination can legitimately appear
        elsewhere for unrelated purposes (e.g. an integrated sensor reused
        to source an ancillary met variable), which is not the flux system
        and must not be compared against it.

        Split-sensor sites declare `Fco2`/`Fe` as a `{sonic_anemometer,
        irga}` dict (`Fh` is conventionally single-form — sonic only per
        `CLAUDE.md` — and is not cross-checked against the pair). Integrated-
        sensor sites (one physical instrument serves both roles) declare all
        three as the same plain string.

        Agreement is per period: a site stitched across a flux-system change
        (e.g. CSAT3 + LI-7500A, then CSAT3B + LI-7500RS) has one pairing per
        era, and only inputs whose begin/end periods overlap must agree.
        """
        dict_pairs: list[tuple[tuple[str, str], InputVariableConfig]] = []
        string_instruments: list[tuple[str, InputVariableConfig]] = []

        for var_name, var_cfg in self.variables.items():
            if var_name not in self.flux_quantities:
                continue
            for input_cfg in var_cfg.input_variables:
                inst = input_cfg.instrument
                if isinstance(inst, dict):
                    if "sonic_anemometer" in inst and "irga" in inst:
                        dict_pairs.append(
                            ((inst["sonic_anemometer"], inst["irga"]), input_cfg)
                        )
                else:
                    string_instruments.append((inst, input_cfg))

        # The flux system can change at a logger changeover, so the
        # instruments only have to agree where the inputs' periods overlap.
        if dict_pairs:
            _check_agree_where_overlapping(
                dict_pairs, "inconsistent sonic_anemometer/irga pairings"
            )
        else:
            _check_agree_where_overlapping(
                string_instruments, "inconsistent single-instrument names"
            )

        return self


def _check_agree_where_overlapping(
    entries: list[tuple[object, InputVariableConfig]], problem: str
) -> None:
    """Raise if two inputs with overlapping periods name different instruments."""
    for i, (first, first_cfg) in enumerate(entries):
        for second, second_cfg in entries[i + 1 :]:
            if first != second and _periods_overlap(first_cfg, second_cfg):
                raise ValueError(
                    f"Fco2/Fh/Fe declare {problem} for the same period: "
                    f"{first} ({_period(first_cfg)}) vs {second} "
                    f"({_period(second_cfg)})"
                )


def validate_all(config_data: dict) -> tuple[bool, list]:
    """External config validator, e.g. for testing UI-entered config validity."""
    try:
        SiteConfig(**config_data)
        return True, []
    except ValidationError as e:
        return False, e.errors()


def validate_L1_config_structure(file: Path | str) -> SiteConfig:
    """Validate YAML structure and return Config object."""
    return SiteConfig(**read_yml(file_path=file, enforce_unique_keys=True))
