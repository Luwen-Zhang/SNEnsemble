import os

__root__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

__all__ = ["data", "model", "trainer", "utils"]

setting = dict(
    verbose_per_epoch=20,
)
