"""fluxcore: eddy covariance processing core (raw logger files -> L1/L2 Zarr).

The names below are the supported API: entry points, contract objects, and
introspection of fluxcore's own domain. Anything reached by a deeper import
(fluxcore.<subpackage>...) is internal and may change without notice.
"""

from fluxcore.domain.constants import DATA_TIME_FORMAT, SITE_PLACEHOLDER
from fluxcore.domain.data_models.metadata_classes import (
    CanonicalQuantityMetadata,
    SiteMetadata,
)
from fluxcore.domain.enums import (
    DEFAULT_FILE_FORMAT,
    DiagnosticType,
    FileFormat,
    FluxSystemType,
    LoggerType,
    StatisticType,
    VariableType,
)
from fluxcore.orchestration import (
    build_L1_nc,
    build_L1_zarr,
    build_L2_nc,
    build_L2_zarr,
)
from fluxcore.orchestration.dataframe_builder import build_dataframe_from_context
from fluxcore.orchestration.dataset_builder import build_dataset_from_context
from fluxcore.services.data import profile_storage
from fluxcore.services.data.calculations import get_calculation
from fluxcore.services.data.raw_data_loader import (
    get_data_loader,
    load_raw_data,
    load_raw_data_since,
    load_raw_header,
    peek_last_timestamp,
)
from fluxcore.services.metadata.canonical_quantity_registry import (
    build_canonical_quantity_registry,
)
from fluxcore.services.metadata.file_group_builder import (
    FileGroup,
    get_variables_from_file,
)
from fluxcore.services.metadata.qc_config_schema import (
    RangeCheckSpec,
    SiteQCConfig,
    VariableQCSpec,
    load_dependency_defaults,
    load_qc_config,
    load_range_defaults,
    lookup_default_range,
    resolve_default_range,
    resolve_qc_config,
)
from fluxcore.services.metadata.runtime_config_builder import (
    SiteRuntimeConfig,
    build_runtime_config,
    build_runtime_config_from_file,
)
from fluxcore.services.metadata.site_config_schema import (
    SiteConfig,
    validate_all,
    validate_L1_config_structure,
)
from fluxcore.services.metadata.site_metadata_builder import (
    SiteContext,
    build_site_metadata_from_file,
)
from fluxcore.services.metadata.variable_name_parser import parse_variable_name
from fluxcore.services.metadata.variable_registry import VariableSpec

__all__ = [
    # Level builders (modules): build/update a Zarr store, export NetCDF
    "build_L1_nc",
    "build_L1_zarr",
    "build_L2_nc",
    "build_L2_zarr",
    "build_dataframe_from_context",
    "build_dataset_from_context",
    # Contract objects
    "CanonicalQuantityMetadata",
    "FileGroup",
    "RangeCheckSpec",
    "SiteConfig",
    "SiteContext",
    "SiteMetadata",
    "SiteQCConfig",
    "SiteRuntimeConfig",
    "VariableQCSpec",
    "VariableSpec",
    # Assembling contract objects from configs
    "build_runtime_config",
    "build_runtime_config_from_file",
    "build_site_metadata_from_file",
    "load_qc_config",
    "validate_L1_config_structure",
    "validate_all",
    # Domain vocabulary
    "DATA_TIME_FORMAT",
    "DEFAULT_FILE_FORMAT",
    "DiagnosticType",
    "FileFormat",
    "FluxSystemType",
    "LoggerType",
    "SITE_PLACEHOLDER",
    "StatisticType",
    "VariableType",
    # Domain introspection
    "build_canonical_quantity_registry",
    "get_calculation",
    "load_dependency_defaults",
    "load_range_defaults",
    "lookup_default_range",
    "parse_variable_name",
    "profile_storage",
    "resolve_default_range",
    "resolve_qc_config",
    # Reading raw files
    "get_data_loader",
    "get_variables_from_file",
    "load_raw_data",
    "load_raw_data_since",
    "load_raw_header",
    "peek_last_timestamp",
]
