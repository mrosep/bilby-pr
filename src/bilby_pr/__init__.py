"""Posterior repartitioning in bilby."""
import os
from importlib.metadata import PackageNotFoundError, version

from bilby.core.utils import logger

# The margarine flows need Keras 2 (tf_keras); this must be set before tensorflow is imported
if os.environ.get("TF_USE_LEGACY_KERAS", "1") not in ("1", "true", "True"):
    logger.warning(
        f"Overriding TF_USE_LEGACY_KERAS={os.environ['TF_USE_LEGACY_KERAS']!r} with '1': "
        "the margarine flows need Keras 2 (tf_keras)"
    )
os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
