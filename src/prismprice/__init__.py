"""
PrismPrice — dynamic pricing decision support.

Optimises for long-run contribution profit rather than single-transaction
margin. See the README for the objective function and layer architecture.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("prismprice")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
