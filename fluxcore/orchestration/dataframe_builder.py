#!/usr/bin/env python3
"""Build canonical DataFrames from raw instrument file groups.

Public API
----------
build_dataframe_from_context(ctx, quantities, start_date) -> pd.DataFrame
build_dataframe_and_specs(ctx, quantities, start_date)
    -> (pd.DataFrame, dict[str, VariableSpec])
build_dataframe(file_groups, specs_by_alias, n_samples, start_date,
                flux_file, time_step, quantities)
    -> pd.DataFrame
"""

from collections.abc import Callable

import pandas as pd

from fluxcore.domain.enums import DiagnosticType, StatisticType, VariableType
from fluxcore.infrastructure.data_conditioning import condition_dataframe
from fluxcore.services.data import raw_data_loader, unit_conversions
from fluxcore.services.metadata.canonical_quantity_registry import (
    resolve_variance_units,
)
from fluxcore.services.metadata.file_group_builder import (
    FileGroup,
    build_file_groups,
)
from fluxcore.services.metadata.site_metadata_builder import SiteContext
from fluxcore.services.metadata.variable_registry import (
    VariableSpec,
    build_variable_registry,
    canonical_output_name,
    group_by_canonical_name,
)


def build_dataframe_from_context(
    ctx: SiteContext,
    quantities: set[str] | None = None,
    start_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a canonical DataFrame from a fully-assembled site context.

    Assembles file_groups and specs_by_alias from the context, then delegates to
    build_dataframe. All site-specific parameters (n_samples, flux_file,
    time_step) are sourced from the context.

    Args:
        ctx: Site runtime config and metadata.
        quantities: Optional set of base quantity names to load (e.g.
            ``{'Fco2', 'Fh'}``). When None, all configured variables are built.
        start_date: If provided, records before this timestamp are discarded.

    Returns:
        Canonical DataFrame with DatetimeIndex.
    """
    df, _ = build_dataframe_and_specs(ctx, quantities=quantities, start_date=start_date)
    return df


def build_dataframe_and_specs(
    ctx: SiteContext,
    quantities: set[str] | None = None,
    start_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, VariableSpec]]:
    """As build_dataframe_from_context, also returning the alias-keyed specs.

    The specs are the full set built from the context (not filtered by
    `quantities`); dataset_builder derives variable attrs from them.
    """
    runtime_cfg = ctx.runtime_config
    file_groups = build_file_groups(runtime_cfg)
    for group in file_groups.values():
        group.validate_or_raise()
    specs_by_alias = build_variable_registry(
        runtime_cfg=runtime_cfg, file_groups=file_groups
    )

    df = build_dataframe(
        file_groups=file_groups,
        specs_by_alias=specs_by_alias,
        quantities=quantities,
        n_samples=ctx.metadata.n_samples,
        start_date=start_date,
        flux_file=runtime_cfg.flux_file,
        time_step=ctx.metadata.time_step,
    )
    return df, specs_by_alias


def build_dataframe(
    file_groups: dict[str, FileGroup],
    specs_by_alias: dict[str, VariableSpec],
    n_samples: int | None = None,
    start_date: pd.Timestamp | None = None,
    flux_file: str | None = None,
    time_step: int | None = None,
    quantities: set[str] | None = None,
) -> pd.DataFrame:
    """Build the canonical dataframe.

    When ``quantities`` is supplied, the specs_by_alias is filtered to only entries
    where ``spec.quantity in quantities``, and ``file_groups`` is further
    reduced to only those groups referenced by the filtered specs_by_alias.  When
    ``None``, the full specs_by_alias and all file groups are used.

    Step 1 — per file group: load all files, apply aliasing, conversions,
              condition, and concatenate vertically.
    Step 2 — concatenate all file groups horizontally.
    Step 3 — merge overlapping instrument periods into single columns.
    Step 4 — rename aliased columns to canonical output names.
    Step 5 — truncate to the flux record's temporal extent, so that
              ancillary data predating the flux system does not produce empty
              output years. The flux record is ``flux_file`` (the current
              flux file) plus every earlier file stitched into the same
              variables (see _flux_extent_groups), so a logger changeover
              doesn't cut off the older era. With ``start_date`` (an
              incremental tail), the tail instead spans start_date to the last
              new record in ``flux_file`` itself on the regular time grid, and
              is empty if there is no new flux record: exactly the rows a full
              build would add after start_date, so appending it can't run the
              store ahead of the flux data or leave holes in its time axis.
    """
    extent_groups = _flux_extent_groups(specs_by_alias, flux_file)

    if quantities is not None:
        specs_by_alias = {
            alias: spec
            for alias, spec in specs_by_alias.items()
            if spec.quantity in quantities
        }
        active_groups = {spec.file_group for spec in specs_by_alias.values()}
        file_groups = {
            name: fg for name, fg in file_groups.items() if name in active_groups
        }

    # Step 1
    empty = pd.DataFrame(index=pd.DatetimeIndex([], name="time"))
    dfs = []
    flux_index = None
    extent_indexes = []
    for group_name, file_group in file_groups.items():
        loader = raw_data_loader.get_data_loader(
            file_group.file_format, start_date=start_date
        )
        group_df = _build_file_group_dataframe(
            file_group=file_group,
            loader=loader,
            specs_by_alias=specs_by_alias,
            n_samples=n_samples,
            start_date=start_date,
            time_step=time_step,
        )
        if not group_df.columns.empty:
            # Only trust flux_index from a group that actually has rows
            # this cycle — an empty (NaN-placeholder) flux group must not
            # set it to an empty index, which would break Step 5 below.
            if group_name == flux_file and not group_df.empty:
                flux_index = group_df.index
            if group_name in extent_groups and not group_df.empty:
                extent_indexes.append(group_df.index)
            dfs.append(group_df)

    # Step 2
    if not dfs:
        # No file group had any rows in the requested window (e.g. a
        # checkpoint-based tail read with no new records since last run) —
        # nothing to concatenate. Return an empty frame with a properly
        # typed time index rather than letting pd.concat raise, so callers
        # can treat "no new data yet" as a normal, empty result.
        return empty
    df = pd.concat(dfs, axis=1)

    # Step 3
    df = _merge_overlapping_variables(df=df, specs_by_alias=specs_by_alias)

    # Step 4
    df = _rename_to_canonical(df=df, specs_by_alias=specs_by_alias)

    # Step 5
    if start_date is None:
        # Only when the current flux file contributed rows: a quantities
        # filter that leaves out the flux file (or an empty one) skips the
        # trim, as before, rather than trimming to an older era alone.
        if flux_index is not None:
            start = min(index.min() for index in extent_indexes)
            end = max(index.max() for index in extent_indexes)
            df = df.loc[start:end]
    elif flux_file in file_groups:
        if flux_index is None:
            return empty
        df = df.loc[: flux_index.max()]
        if time_step is not None:
            df = df.reindex(
                pd.date_range(
                    start_date,
                    flux_index.max(),
                    freq=f"{time_step}min",
                    name=df.index.name,
                )
            )

    return df


def _flux_extent_groups(
    specs_by_alias: dict[str, VariableSpec], flux_file: str | None
) -> set[str]:
    """File groups whose records make up the flux record.

    ``flux_file`` itself, plus every group that supplies an earlier (or
    later) era of a variable ``flux_file`` supplies — i.e. the old logger's
    files after a changeover, stitched in through the variables' raw inputs.
    A site that has never changed logger gets just ``{flux_file}``.
    Ancillary groups (soil, profile) that feed only their own variables are
    not included, so they are still trimmed to the flux record.
    """
    if flux_file is None:
        return set()
    flux_variables = {
        spec.canonical_name
        for spec in specs_by_alias.values()
        if spec.file_group == flux_file
    }
    return {flux_file} | {
        spec.file_group
        for spec in specs_by_alias.values()
        if spec.canonical_name in flux_variables
    }


def _build_file_group_dataframe(
    file_group: FileGroup,
    loader: Callable,
    specs_by_alias: dict[str, VariableSpec],
    n_samples: int | None = None,
    start_date: pd.Timestamp | None = None,
    time_step: int | None = None,
) -> pd.DataFrame:
    """Load and process all files in a file group.

    For each file: load, filter to expected variables, copy each raw column
    to its inputs' aliases, apply statistical transforms and unit
    conversions.  Then concatenate
    vertically across the group's files and condition the time index.
    Files whose latest record predates start_date are skipped entirely.

    Overlapping timestamps cascade in `FileGroup.all_files` order: master,
    then backups newest to oldest, then patches. Each file only contributes
    the timestamps no higher-precedence file has; the whole record is taken
    from the winning file (records are not merged column by column).
    """
    # Each input gets its own copy of its raw column: one column can feed
    # several inputs (eras of a column whose units or instrument changed,
    # or two variables), so columns are copied per alias, not renamed.
    aliases_by_raw_name: dict[str, list[str]] = {}
    for entry in specs_by_alias.values():
        if entry.file_group == file_group.group:
            aliases_by_raw_name.setdefault(entry.raw_name, []).append(entry.alias)
    group_aliases = [a for aliases in aliases_by_raw_name.values() for a in aliases]

    dfs = []
    for file in file_group.variables_by_file:
        df = loader(file)

        if start_date is not None:
            if df.empty or df.index.max() < start_date:
                continue
            df = df[df.index >= start_date]

        df = _filter_variables(df=df, variables=file_group.expected_variables)

        df = pd.DataFrame(
            {
                alias: df[col]
                for col in df.columns
                for alias in aliases_by_raw_name.get(col, [])
            },
            index=df.index,
        )

        df = _apply_conversions(
            df=df, specs_by_alias=specs_by_alias, n_samples=n_samples
        )

        dfs.append(df)

    if not dfs:
        # No file in this group contributed a row in the requested window
        # (e.g. this group's source file simply hasn't been touched since
        # the last incremental checkpoint — a slower-cadence logger, say).
        # Still register the group's expected columns as an empty frame
        # rather than dropping them: pd.concat(axis=1) in build_dataframe
        # then NaN-fills them across whatever time range other groups DID
        # produce, keeping this group's variables in the result at all —
        # a real gap, not a silent omission. This matters most for Zarr's
        # incremental append: without it, a variable that's absent for one
        # tail cycle never gets its store array appended to at all, and
        # falls permanently behind the variables that did have new rows.
        return pd.DataFrame(
            columns=group_aliases,
            index=pd.DatetimeIndex([], name="time"),
            dtype="float64",
        )

    # dfs is in precedence order, so keep="first" keeps the winning file's
    # record; dedupe before sorting because sort_index isn't stable.
    df = pd.concat(dfs)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # Same rule as the no-files branch above, for a configured variable whose
    # column is absent from every file read this time (e.g. a sensor dropped
    # from the logger program, so only older backup files still carry it): add
    # it as all-NaN. The group's columns must always be its full configured
    # set, independent of which files a given read happened to touch;
    # otherwise an incremental tail's variable set differs from the store's
    # (built from all files) and the append is refused.
    missing = [alias for alias in group_aliases if alias not in df.columns]
    if missing:
        df = df.reindex(columns=[*df.columns, *missing])

    return condition_dataframe(df, interval_out=time_step)


def _filter_variables(df: pd.DataFrame, variables: set) -> pd.DataFrame:
    """Keep only columns present in the expected variable set."""
    return df[[c for c in df.columns if c in variables]].copy()


def _apply_conversions(
    df: pd.DataFrame,
    specs_by_alias: dict[str, VariableSpec],
    n_samples: int | None = None,
) -> pd.DataFrame:
    """Apply statistical transforms and unit conversions in a single pass.

    For variance inputs (the input's raw statistic, which may differ from the
    variable's): take the square root (data → stdev), then derive the
    post-transform units and convert further if site stdev-form units differ
    from canonical stdev-form units.

    For counter variables: output is always invalid_count (0 = no error).
    Sites storing valid_count are converted via unit_conversions; sites
    already storing invalid_count pass through unchanged. diag_type on the
    VariableSpec determines which path is taken.

    For all other variables: convert units if site units differ from canonical.

    QC variables (dimensionless on both sides) pass through unchanged.
    """
    for variable in df.columns:
        if variable not in specs_by_alias:
            continue

        spec = specs_by_alias[variable]
        from_units = spec.site_units

        # The raw data's own statistic, not the variable's: a stitched
        # variable can mix variance and standard-deviation inputs.
        if spec.raw_statistic_type == StatisticType.VAR:
            df[variable] = df[variable] ** 0.5
            from_units = resolve_variance_units(spec.site_units)

        if spec.variable_type == VariableType.COUNTER:
            if spec.diag_type == DiagnosticType.VALID_COUNT:
                if n_samples is None:
                    raise ValueError(
                        f"Counter variable {variable!r} has diag_type="
                        f"'valid_count' but n_samples is not available. "
                        f"Ensure site metadata includes time_step and freq_hz."
                    )
                converter = unit_conversions.get_unit_conversion("Diag")
                try:
                    result = converter(
                        data=df[variable],
                        from_units="valid_count",
                        n_samples=n_samples,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Diagnostic conversion failed for {variable!r}"
                    ) from e
                df[variable] = result
            # INVALID_COUNT: pass through unchanged
            continue

        if from_units != spec.canonical_units:
            converter = unit_conversions.get_unit_conversion(spec.quantity)
            try:
                result = converter(data=df[variable], from_units=from_units)
            except Exception as e:
                raise RuntimeError(
                    f"Unit conversion failed for {variable!r} "
                    f"({from_units!r} → {spec.canonical_units!r})"
                ) from e
            df[variable] = result

    return df


def _get_merge_blocks(specs_by_alias: dict[str, VariableSpec]) -> dict:
    """Identify canonical variables constructed from multiple raw inputs.

    These are instrument changeovers.

    Returns:
        dict keyed by canonical name; values are alias → {begin, end} dicts.
    """
    rslt = {}

    for canonical_name, var_specs in group_by_canonical_name(specs_by_alias).items():
        if len(var_specs) <= 1:
            continue

        rslt[canonical_name] = {
            spec.alias: {"begin": spec.begin, "end": spec.end} for spec in var_specs
        }

    return rslt


def _merge_overlapping_variables(
    df: pd.DataFrame,
    specs_by_alias: dict[str, VariableSpec],
) -> pd.DataFrame:
    """Merge overlapping instrument periods into single canonical columns."""
    merge_blocks = _get_merge_blocks(specs_by_alias)

    if not merge_blocks:
        return df

    merged = {}
    drop_cols = []

    for canonical_name, block in merge_blocks.items():
        merged[canonical_name] = _merge_block(df=df, block=block)
        drop_cols.extend(block.keys())

    return df.drop(columns=drop_cols, errors="ignore").join(pd.DataFrame(merged))


def _merge_block(
    df: pd.DataFrame,
    block: dict[str, dict],
) -> pd.Series:
    """Assemble one canonical series from consecutive instrument segments."""
    result = pd.Series(index=df.index, dtype=float)

    for alias, info in block.items():
        if alias not in df.columns:
            # Alias's source file contributed no rows in the requested
            # window (e.g. a start_date-restricted tail read past this
            # instrument era) — nothing for this segment to contribute.
            continue
        segment = df.loc[info["begin"] : info["end"], alias]
        result.loc[segment.index] = segment

    return result


def _rename_to_canonical(
    df: pd.DataFrame,
    specs_by_alias: dict[str, VariableSpec],
) -> pd.DataFrame:
    """Rename aliased columns to their canonical output names."""
    rename_map = {
        alias: canonical_output_name(spec)
        for alias, spec in specs_by_alias.items()
        if alias in df.columns
    }

    return df.rename(columns=rename_map)
