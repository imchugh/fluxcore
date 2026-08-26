#!/usr/bin/env python3
"""Build FileGroup objects (master path, format, backups) for a site's input files."""

from dataclasses import dataclass, field
from pathlib import Path

from domain.enums import FileType
from infrastructure import file_io
from services.data import raw_data_loader
from services.metadata.core.runtime_config_builder import SiteRuntimeConfig


@dataclass
class FileGroup:
    """A group of files (master + backups) and their expected/found variables.

    Header discovery is lazy.
    """

    group: str
    master: Path
    backups: list[Path]
    file_format: str
    expected_variables: set[str] = field(default_factory=set)
    _variables_by_file_cache: dict[Path, set[str]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def all_files(self) -> list[Path]:
        """Master file followed by its backups."""
        return [self.master, *self.backups]

    @property
    def variables_by_file(self) -> dict[Path, set[str]]:
        """Lazy evaluation: read headers only on first access and cache results."""
        if not self._variables_by_file_cache:
            for file in self.all_files:
                header_vars = get_variables_from_file(
                    file_path=file, system_type=self.file_format
                )
                self._variables_by_file_cache[file] = set(header_vars)
        return self._variables_by_file_cache

    def validate(self) -> dict[str, set[str]]:
        """Compare expected variables vs discovered variables.

        Returns:
            Dict with 'found' and 'missing' sets.
        """
        found = set().union(*self.variables_by_file.values())
        missing = self.expected_variables - found
        return {"found": found & self.expected_variables, "missing": missing}

    def validate_or_raise(self) -> None:
        """Raise if any expected variable is missing from the discovered headers."""
        missing = self.validate()["missing"]
        if missing:
            raise ValueError(
                f"File group '{self.group}': expected variable(s) "
                f"{sorted(missing)} not found in {[str(f) for f in self.all_files]}"
            )

    def files_for_variable(self, variable: str) -> list[Path]:
        """Return the list of files in which the variable was found."""
        return [f for f, vars in self.variables_by_file.items() if variable in vars]


def get_variables_from_file(
    file_path: Path | str, system_type: str, incl_backups: bool = False
):
    """Get the variable names from the file header."""
    # Set FileType
    ftype = FileType[system_type]

    # Sort out paths
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist!")

    # Build iterable
    file_list = [file_path]
    if incl_backups:
        file_list.extend(file_io.get_backup_files(file_path=file_path))

    # Load adapter
    header_adapter = raw_data_loader.get_header_adapter(system_type=ftype.name)

    # Iterate over files
    rslt = set()
    for file in file_list:
        rslt.update(header_adapter(file)["variable"])

    # Return a list
    return sorted(rslt)


def build_file_groups(runtime_cfg: SiteRuntimeConfig) -> dict[str, FileGroup]:
    """Build FileGroup objects for all variable groups in the runtime config.

    Raises:
        ValueError: if runtime_cfg.input_data_path is None — this function
            needs to know where the site's raw files live, but has no
            opinion of its own about where that is; the caller (typically
            a TERN-adapter loader) must have resolved and set it.
    """
    if runtime_cfg.input_data_path is None:
        raise ValueError(
            f"Cannot build file groups for site '{runtime_cfg.site_name}': "
            "no input_data_path set on this SiteRuntimeConfig."
        )
    base_path = runtime_cfg.input_data_path
    groups: dict[str, FileGroup] = {}

    for var_def in runtime_cfg.variables.values():
        for raw_var in var_def.raw_inputs:
            group_name = raw_var.file
            file_format = runtime_cfg.get_file_format(group_name)
            file_ext = FileType[file_format].extension
            group = groups.get(group_name)
            if group is None:
                master = base_path / f"{group_name}.{file_ext}"
                backups = file_io.get_backup_files(
                    file_path=master,
                    abs_path=True,
                )
                group = FileGroup(
                    group=group_name,
                    master=master,
                    backups=backups,
                    file_format=file_format,
                )
                groups[group_name] = group

            group.expected_variables.add(raw_var.raw_name)

    return groups
