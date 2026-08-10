#!/usr/bin/env python3
"""Resolve and load internal config files (yml/txt/csv) by name."""

import pathlib

import pandas as pd

from infrastructure.file_io import read_csv, read_text, read_yml
from infrastructure.paths import CONFIG_FILE, CONFIG_PATH

ConfigType = dict | str | pd.DataFrame
ALLOWED_CONFIG_TYPES = [".yml", ".txt", ".csv"]


def list_internal_config_files() -> list[pathlib.Path]:
    """List the absolute paths of the available internal configuration files.

    Note that paths.yml is not available here — it is consumed at
    infrastructure level by the paths module.

    Returns:
        List of config file paths.
    """
    return [
        file
        for file in CONFIG_PATH.glob("*")
        if file.name != CONFIG_FILE and file.suffix in ALLOWED_CONFIG_TYPES
    ]


def list_internal_config_names() -> list:
    """List the names of the available internal configuration files.

    Returns:
        List of config names.
    """
    return list(_build_config_index().keys())


def get_file_from_config_name(name: str) -> pathlib.Path:
    """Return the absolute path to the file given the configuration name.

    Args:
        name: configuration name.

    Raises:
        FileNotFoundError: raised if no file with compatible name is found.

    Returns:
        Absolute path to the config file.
    """
    index = _build_config_index()
    try:
        return index[name]
    except KeyError:
        raise FileNotFoundError(f"No file with name {name} found") from None


def _build_config_index() -> dict[str, pathlib.Path]:
    """Map config name to file.

    Returns:
        Dict containing config names as keys, absolute paths as values.
    """
    return {file.stem: file for file in list_internal_config_files()}


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


def load_config_file_from_name(name: str) -> ConfigType:
    """Load the configuration file via configuration name.

    Args:
        name: configuration name.

    Returns:
        Parsed config content — dict (yml), str (txt), or DataFrame (csv).
    """
    file = get_file_from_config_name(name=name)
    return load_config_file(file=file)
