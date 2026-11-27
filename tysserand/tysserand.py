"""
Deprecated namespace: ``from tysserand import tysserand as ty``.

The code lives in the submodules of the package and every public function is
available from the package itself: use ``import tysserand as ty``.
This module will be removed in a future release.
"""

import warnings

# FutureWarning rather than DeprecationWarning: the latter is hidden by default
# when triggered from library code, and users mostly import from notebooks/scripts.
warnings.warn(
    "`from tysserand import tysserand as ty` is deprecated and will be removed "
    "in a future release, use `import tysserand as ty` instead.",
    FutureWarning,
    stacklevel=2,
)

from .generators import *  # noqa: E402,F401,F403
from .convert import *  # noqa: E402,F401,F403
from .io import *  # noqa: E402,F401,F403
from .distances import *  # noqa: E402,F401,F403
from .build import *  # noqa: E402,F401,F403
from .contacting import *  # noqa: E402,F401,F403
from .plot import *  # noqa: E402,F401,F403
from .scoring import *  # noqa: E402,F401,F403
from .interop import *  # noqa: E402,F401,F403
from .annotation import *  # noqa: E402,F401,F403
