"""
Backward-compatible namespace: ``from tysserand import tysserand as ty``.

The code now lives in the submodules of the package, use ``import tysserand as ty``
in new code.
"""

from .generators import *
from .convert import *
from .io import *
from .distances import *
from .build import *
from .contacting import *
from .plot import *
from .scoring import *
from .interop import *
from .annotation import *
