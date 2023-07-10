import os
import numpy as np
import warnings

np.int = int  # ``np.int`` is a deprecated alias for the builtin ``int``.

__root__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

__all__ = ["data", "model", "trainer", "config", "utils"]

setting = dict(
    # The random seed for numpy and pytorch (including cuda).
    random_seed=42,
    # If the memory of the system (gpu or cpu) is lower than 6 GiBs, set ``low_memory`` to True.
    # TODO: Enlarge bayes search space when low_memory is set to False.
    low_memory=True,
    verbose_per_epoch=20,
    # To save memory, turn test_with_no_grad to True. However, this operation will make
    # some models that need gradients within the loss function invalid.
    test_with_no_grad=True,
    # Debug mode might change behaviors of models. By default, epoch will be set to 2, n_calls to minimum, and
    # bayes_epoch to 1.
    debug_mode=False,
    # If batch_size // len(training set) < limit_batch_size, the batch_size is forced to be len(training set) to avoid
    # potential numerical issue. For Tabnet, this is extremely important because a small batch may cause NaNs and
    # further CUDA device-side assert in the sparsemax function. Set to -1 to turn off this check (NOT RECOMMENDED!!).
    limit_batch_size=6,
    # Default paths to configure trainers, data modules, and models.
    default_output_path="output",
    default_config_path="configs",
    default_data_path="data",
)

if setting["debug_mode"]:
    warnings.warn("The debug mode is activated. Please confirm whether it is desired.")

if setting["limit_batch_size"] == -1:
    warnings.warn(
        "limit_batch_size is disabled, which is not recommended. A very small batch may cause unexpected "
        "numerical issue, especially for TabNet."
    )


def check_grad_in_loss():
    if setting["test_with_no_grad"]:
        return False
    return True
