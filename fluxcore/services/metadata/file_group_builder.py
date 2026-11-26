#!/usr/bin/env python3
"""Build FileGroup objects (master, backups, patches) for a site's input files."""

from dataclasses import dataclass, field
from pathlib import Path

from fluxcore.domain.enums import DEFAULT_FILE_FORMAT, FileFormat, LoggerType
from fluxcore.infrastructure import file_io
from fluxcore.services.data import raw_data_loader
from fluxcore.services.metadata.runtime_config_builder import SiteRuntimeConfig


@dataclass
class FileGroup:
    """A group of files (master, backups, patches) and their variables.

    `backups` and `patches` are oldest first, as numbered on disk (see
    `file_io.get_numbered_files`). Patches are gap-fill records for the
    master, kept outside the raw-data directory so raw files are never
    edited. Header discovery is lazy.
    """

    group: str
    master: Path
    backups: list[Path]
    logger_type: LoggerType
    expected_variables: set[str] = field(default_factory=set)
    patches: list[Path] = field(default_factory=list)
    _variables_by_file_cache: dict[Path, set[str]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def file_format(self) -> FileFormat:
        """File format this group's files are read as."""
        return DEFAULT_FILE_FORMAT[self.logger_type]

    @property
    def all_files(self) -> list[Path]:
        """Every file in the group, in precedence order (highest first).

        Master, then backups newest to oldest, then patches newest to
        oldest. Where files share a timestamp, the record from the earliest
        file in this list wins; each later file only contributes timestamps
        none before it covers. So newer logger data beats older, and a patch
        only ever fills gaps left by the logger's own files.
        """
        return [self.master, *reversed(self.backups), *reversed(self.patches)]

    @property
    def variables_by_file(self) -> dict[Path, set[str]]:
        """Lazy evaluation: read headers only on first access and cache results."""
        if not self._variables_by_file_cache:
            for file in self.all_files:
                header_vars = get_variables_from_file(
                    file_path=file, file_format=self.file_format
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


def get_variables_from_file(
    file_path: Path | str, file_format: FileFormat | str
) -> list[str]:
    """Return the sorted variable names from a raw file's header."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist!")

    header = raw_data_loader.load_raw_header(file_path, file_format)
    return sorted(set(header["variable"]))


def build_file_groups(runtime_cfg: SiteRuntimeConfig) -> dict[str, FileGroup]:
    """Build FileGroup objects for all variable groups in the runtime config.

    Raises:
        ValueError: if runtime_cfg.input_data_path is None — this function
            needs to know where the site's raw files live, but has no
            opinion of its own about where that is; the caller (typically
            a network adapter) must have resolved and set it.
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
            if raw_var.calculate_from is not None:
                continue  # a calculated input reads no file
            group_name = raw_var.file
            group = groups.get(group_name)
            if group is None:
                master = base_path / runtime_cfg.get_filename(group_name)
                backups = file_io.get_backup_files(
                    file_path=master,
                    abs_path=True,
                )
                patches = (
                    file_io.get_numbered_files(
                        file_path=master,
                        suffix="patch",
                        directory=runtime_cfg.input_patch_path,
                    )
                    if runtime_cfg.input_patch_path is not None
                    else []
                )
                group = FileGroup(
                    group=group_name,
                    master=master,
                    backups=backups,
                    patches=patches,
                    logger_type=runtime_cfg.get_logger_type(group_name),
                )
                groups[group_name] = group

            group.expected_variables.add(raw_var.raw_name)

    return groups
