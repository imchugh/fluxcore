#!/usr/bin/env python3
"""Resolve and load internal config files (yml/txt/csv) by name.

Generic core. CONFIG_PATH is the core's own config directory (this repo's
configs/, which holds only network-agnostic files). Every lookup takes an
optional config_dir, so an adapter can load its own config files through the
same functions: TERN's live in configs/tern/ (infrastructure.tern.paths
.CONFIG_PATH), which the default, non-recursive lookup never sees.
"""

import pathlib

import pandas as pd

from infrastructure.file_io import read_csv, read_text, read_yml

CONFIG_PATH = pathlib.Path(__file__).parents[1] / "configs"

ConfigType = dict | str | pd.DataFrame
ALLOWED_CONFIG_TYPES = [".yml", ".txt", ".csv"]


def list_internal_config_files(
    config_dir: pathlib.Path = CONFIG_PATH,
) -> list[pathlib.Path]:
    """List the absolute paths of the configuration files in config_dir.

    Args:
        config_dir: directory to list (not recursive). Defaults to the core's
            own CONFIG_PATH.

    Returns:
        List of config file paths.
    """
    return [
        file
        for file in pathlib.Path(config_dir).glob("*")
        if file.suffix in ALLOWED_CONFIG_TYPES
    ]


def list_internal_config_names(config_dir: pathlib.Path = CONFIG_PATH) -> list:
    """List the names of the configuration files in config_dir.

    Returns:
        List of config names.
    """
    return list(_build_config_index(config_dir).keys())


def get_file_from_config_name(
    name: str, config_dir: pathlib.Path = CONFIG_PATH
) -> pathlib.Path:
    """Return the absolute path to the file given the configuration name.

    Args:
        name: configuration name.
        config_dir: directory to look in. Defaults to CONFIG_PATH.

    Raises:
        FileNotFoundError: raised if no file with compatible name is found.

    Returns:
        Absolute path to the config file.
    """
    index = _build_config_index(config_dir)
    try:
        return index[name]
    except KeyError:
        raise FileNotFoundError(
            f"No file with name {name} found in {config_dir}"
        ) from None


def _build_config_index(config_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """Map config name to file.

    Returns:
        Dict containing config names as keys, absolute paths as values.
    """
    return {file.stem: file for file in list_internal_config_files(config_dir)}


def load_config_file(file: pathlib.Path | str) -> ConfigType:
    """Load the configuration file.

    Args:
        file: absolute path of file.

    Raises:
        ValueError: raised if unsupported file suffix passed.

    Returns:
        Parsed config content — dict (yml), str (txt), or DataFrame (csv).
    """
    file = pathlib.Path(file)
    if file.suffix == ".txt":
        return read_text(file_path=file)
    if file.suffix == ".yml":
        return read_yml(file_path=file)
    if file.suffix == ".csv":
        return read_csv(file_path=file)
    raise ValueError(f"Unsupported config file type: {file.suffix}")


def load_config_file_from_name(
    name: str, config_dir: pathlib.Path = CONFIG_PATH
) -> ConfigType:
    """Load the configuration file via configuration name.

    Args:
        name: configuration name.
        config_dir: directory to look in. Defaults to CONFIG_PATH.

    Returns:
        Parsed config content — dict (yml), str (txt), or DataFrame (csv).
    """
    file = get_file_from_config_name(name=name, config_dir=config_dir)
    return load_config_file(file=file)
