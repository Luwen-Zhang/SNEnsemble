import os

__root__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

__all__ = ["data", "model", "trainer", "utils"]

setting = dict(
    # If the memory of the system (gpu or cpu) is lower than 6 GiBs, set `low_memory` to True.
    # TODO: Enlarge bayes search space when low_memory is set to False.
    low_memory=True,
    verbose_per_epoch=20,
    # To save memory, turn test_with_no_grad to True and input_requires_grad to False. However, this operation will make
    # some models that need gradients within the loss function invalid.
    test_with_no_grad=False,
    input_requires_grad=True,
    # Debug mode might change behaviors of models. By default, epoch will be set to 2, n_calls to 11, and
    # bayes_epoch to 1.
    debug_mode=False,
)


def check_grad_in_loss():
    if setting["test_with_no_grad"] or not setting["input_requires_grad"]:
        return False
    return True
