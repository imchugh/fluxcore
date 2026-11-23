#!/usr/bin/env python3
"""Standalone, file-in entry points: one control file per site per level.

Front door 2 (PyFluxPro-style). Each function only unpacks its control file
into the same contract objects a network adapter builds, then calls the
live-object builder (front door 1), so the two routes cannot diverge:

    run_L1(control_file)  site config (+ paths, site_metadata, global_attrs)
                          -> SiteContext -> build_L1_zarr.build/update
    run_L2(control_file)  site QC config (+ paths)
                          -> SiteQCConfig -> build_L2_zarr.build/update

Every directory and object can also be passed as an argument, which wins
over the control file. Relative paths in a control file resolve against
the control file's own directory, so a project folder can be moved whole.
L2 needs no metadata or global attributes: it inherits them from the L1
store it reads.
"""

import pathlib

from fluxcore.orchestration import build_L1_zarr, build_L2_zarr
from fluxcore.services.metadata.qc_config_schema import (
    SiteQCConfig,
    validate_qc_config_structure,
)
from fluxcore.services.metadata.runtime_config_builder import build_runtime_config
from fluxcore.services.metadata.site_config_schema import (
    validate_L1_config_structure,
)
from fluxcore.services.metadata.site_metadata_builder import (
    SiteContext,
    SiteMetadata,
    build_site_metadata,
)

PathLike = pathlib.Path | str


def run_L1(
    control_file: PathLike,
    output_dir: PathLike | None = None,
    input_data_path: PathLike | None = None,
    input_patch_path: PathLike | None = None,
    site_metadata: SiteMetadata | None = None,
    global_attrs: dict | None = None,
    update: bool = False,
) -> pathlib.Path:
    """Build (or update) a site's L1 Zarr store from its L1 control file.

    The control file is the site config, optionally with `paths`
    (`input_data_path`, `input_patch_path`, `output_dir`), `site_metadata`
    and `global_attrs` sections.

    Args:
        control_file: the site's L1 control file.
        output_dir: directory for the store; overrides `paths.output_dir`.
        input_data_path: raw-data directory; overrides `paths.input_data_path`.
        input_patch_path: `.patch` file directory; overrides
            `paths.input_patch_path`. Optional: none means no patches.
        site_metadata: overrides the `site_metadata` section.
        global_attrs: merged over the `global_attrs` section, key by key.
        update: append new records only (build_L1_zarr.update) instead of a
            full rebuild.

    Returns:
        Path to the Zarr store written.

    Raises:
        ValueError: a required directory or the site metadata is in neither
            the arguments nor the control file.
    """
    control_file = pathlib.Path(control_file)
    config = validate_L1_config_structure(file=control_file)
    paths = config.paths

    data_path = _resolve_dir(
        input_data_path,
        paths and paths.input_data_path,
        control_file,
        "input_data_path",
    )
    patch_path = _resolve_dir(
        input_patch_path,
        paths and paths.input_patch_path,
        control_file,
        "input_patch_path",
        required=False,
    )
    out_dir = _resolve_dir(
        output_dir, paths and paths.output_dir, control_file, "output_dir"
    )

    if site_metadata is None:
        if config.site_metadata is None:
            raise ValueError(
                f"{control_file}: no site metadata; add a site_metadata "
                "section or pass site_metadata="
            )
        site_metadata = build_site_metadata(config.site_metadata)

    attrs = {**(config.global_attrs or {}), **(global_attrs or {})} or None

    ctx = SiteContext(
        runtime_config=build_runtime_config(
            config, input_data_path=data_path, input_patch_path=patch_path
        ),
        metadata=site_metadata,
    )
    if update:
        return build_L1_zarr.update(ctx, output_dir=out_dir, global_attrs=attrs)
    return build_L1_zarr.build(ctx, output_dir=out_dir, global_attrs=attrs)


def run_L2(
    control_file: PathLike,
    output_dir: PathLike | None = None,
    l1_dir: PathLike | None = None,
    update: bool = False,
) -> pathlib.Path:
    """Build (or update) a site's L2 Zarr store from its L2 control file.

    The control file is the site's QC config (`site` + `variables`),
    optionally with a `paths` section (`l1_dir`, `output_dir`).

    Args:
        control_file: the site's L2 control file.
        output_dir: directory for the L2 store; overrides `paths.output_dir`.
        l1_dir: directory holding the site's L1 store; overrides
            `paths.l1_dir`.
        update: append new records only (build_L2_zarr.update) instead of a
            full rebuild.

    Returns:
        Path to the L2 Zarr store written.

    Raises:
        ValueError: a required directory is in neither the arguments nor the
            control file.
    """
    control_file = pathlib.Path(control_file)
    schema = validate_qc_config_structure(control_file)
    paths = schema.paths

    l1 = _resolve_dir(l1_dir, paths and paths.l1_dir, control_file, "l1_dir")
    out_dir = _resolve_dir(
        output_dir, paths and paths.output_dir, control_file, "output_dir"
    )

    qc_config = SiteQCConfig(site_name=schema.site, variables=dict(schema.variables))
    if update:
        return build_L2_zarr.update(qc_config, output_dir=out_dir, l1_dir=l1)
    return build_L2_zarr.build(qc_config, output_dir=out_dir, l1_dir=l1)


def _resolve_dir(
    argument: PathLike | None,
    file_value: str | None,
    control_file: pathlib.Path,
    key: str,
    required: bool = True,
) -> pathlib.Path | None:
    """Pick a directory: the argument, else the control file's (relative to it)."""
    if argument is not None:
        return pathlib.Path(argument)
    if file_value:
        return control_file.parent / pathlib.Path(file_value).expanduser()
    if required:
        raise ValueError(
            f"{control_file}: no {key}; set paths.{key} in the control file "
            f"or pass {key}="
        )
    return None
