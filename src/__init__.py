import os
import numpy as np
import warnings

np.int = int  # ``np.int`` is a deprecated alias for the builtin ``int``.

__root__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

__all__ = ["data", "model", "trainer", "config", "utils"]

setting = dict(
    # If the memory of the system (gpu or cpu) is lower than 6 GiBs, set ``low_memory`` to True.
    # TODO: Enlarge bayes search space when low_memory is set to False.
    low_memory=True,
    verbose_per_epoch=20,
    # To save memory, turn test_with_no_grad to True. However, this operation will make
    # some models that need gradients within the loss function invalid.
    test_with_no_grad=False,
    # Debug mode might change behaviors of models. By default, epoch will be set to 2, n_calls to minimum, and
    # bayes_epoch to 1.
    debug_mode=False,
)

if setting["debug_mode"]:
    warnings.warn("The debug mode is activated. Please confirm whether it is desired.")


def check_grad_in_loss():
    if setting["test_with_no_grad"]:
        return False
    return True
