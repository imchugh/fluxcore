#!/usr/bin/env python3
"""Pydantic schema models and structural validation for site YAML configs."""

import re
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

from domain.enums import (
    DiagnosticType,
    FileType,
    FluxSystemType,
    StatisticType,
    VariableType,
)
from infrastructure.file_io import read_yml

VALID_FILE_TYPES = [x.name for x in FileType]


# Input-level configuration


class InputVariableConfig(BaseModel):
    """Schema and validation rules for one raw input variable."""

    model_config = ConfigDict(extra="allow")

    # Define attributes

    # Mandatory
    instrument: str | dict[str, str]
    file: str
    units: str

    # Optional
    diag_type: DiagnosticType | None = None
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


class HorizontalLocationConfig(BaseModel):
    """Schema and validation rules for horizontal instrument location."""

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

    model_config = ConfigDict(extra="allow")

    # Define attributes

    # Mandatory
    horizontal_location: HorizontalLocationConfig


# Output-level variable configuration


class VariableConfig(BaseModel):
    """Schema for one canonical variable's full configuration."""

    model_config = ConfigDict(extra="allow")

    # Define attributes

    # Mandatory
    height: float
    input_variables: dict[str, InputVariableConfig]

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


# Complete site configuration
class SiteConfig(BaseModel):
    """Schema and cross-field validation rules for a complete site config."""

    # Define attributes

    site: str
    file_formats: dict[str, str]
    flux_system: FluxSystemType
    flux_file: str
    variables: dict[str, VariableConfig]

    custom_metadata: CustomMetadataConfig | None = None

    # class-level constants for validation
    diag_prefixes: ClassVar[list[str]] = ["Diag_"]

    @field_validator("file_formats")
    @classmethod
    def validate_file_formats(cls, v: dict[str, str]) -> dict[str, str]:
        """Ensure every file_formats value is a recognised FileType name."""
        invalid = {k: val for k, val in v.items() if val not in VALID_FILE_TYPES}
        if invalid:
            raise ValueError(
                f"file_formats values must be one of {VALID_FILE_TYPES}; "
                f"invalid entries: {invalid}"
            )
        return v

    @model_validator(mode="after")
    def enforce_diag_rules(self):
        """Enforce that counter variables declare a diag_type."""
        for var_name, var_cfg in self.variables.items():
            if var_cfg.variable_type == VariableType.COUNTER:
                for input_name, input_cfg in var_cfg.input_variables.items():
                    if input_cfg.diag_type is None:
                        raise ValueError(
                            f"Counter variable '{var_name}' input "
                            f"'{input_name}' must specify diag_type"
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
            for input_cfg in var_cfg.input_variables.values():
                file_list.add(input_cfg.file)

        if self.flux_file not in file_list:
            raise ValueError("Flux file must be mapped to at least one variable!")

        return self


def validate_variable(config_data: dict, key: str) -> tuple[bool, list]:
    """External variable validator, e.g. for testing UI-entered config validity."""
    try:
        VariableConfig(**config_data["variables"][key])
        return True, []
    except ValidationError as e:
        return False, e.errors()


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
