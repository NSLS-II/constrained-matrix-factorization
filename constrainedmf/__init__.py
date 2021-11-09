from ._version import get_versions
from . import nmf  # noqa: F401
from .nmf.models import NMF  # noqa: F401

__version__ = get_versions()["version"]
del get_versions
