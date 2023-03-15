import os

__root__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

__all__ = ["data", "model", "trainer", "utils"]

setting = dict(
    verbose_per_epoch=20,
    # To save memory, turn test_with_no_grad to True and input_requires_grad to False. However, this operation will make
    # some models that need gradients within the loss function invalid.
    test_with_no_grad=False,
    input_requires_grad=True,
)


def check_grad_in_loss():
    if setting["test_with_no_grad"] or not setting["input_requires_grad"]:
        return False
    return True
