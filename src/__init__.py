import os

__root__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

from src.data import *

__all__ = [
    "data",
    "model",
    "trainer",
]
