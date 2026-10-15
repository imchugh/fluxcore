#!/usr/bin/env python3
"""Load fluxcore's own packaged config files (fluxcore/configs/).

Holds only the core's network-agnostic configs; a network adapter loads its
own config files itself.
"""

import pathlib

from fluxcore.infrastructure.file_io import read_yml

CONFIG_PATH = pathlib.Path(__file__).parents[1] / "configs"


def load_config(name: str) -> dict:
    """Load configs/{name}.yml.

    Raises:
        FileNotFoundError: if no config of that name ships with fluxcore.
    """
    file = CONFIG_PATH / f"{name}.yml"
    if not file.exists():
        raise FileNotFoundError(f"No config named {name!r} in {CONFIG_PATH}")
    return read_yml(file_path=file)
