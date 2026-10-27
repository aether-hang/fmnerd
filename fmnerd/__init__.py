"""FM-NeRD: class-shared SIREN with FreD top-k shift modulation.
author: aether
"""

from .dct import build_dct_matrix, dct_2d, idct_2d
from .renderer import ClassSharedSiren, FMNeRDRenderer, ShiftModulationBank
from .synset import FMNeRD

__all__ = [
    "FMNeRD",
    "FMNeRDRenderer",
    "ClassSharedSiren",
    "ShiftModulationBank",
    "build_dct_matrix",
    "dct_2d",
    "idct_2d",
]
