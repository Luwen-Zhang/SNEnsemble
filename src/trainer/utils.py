from .trainer import Trainer


class NoBayesOpt(object):
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.original = None

    def __enter__(self):
        self.original = self.trainer.args["bayes_opt"]
        self.trainer.args["bayes_opt"] = False

    def __exit__(self, *args):
        self.trainer.args["bayes_opt"] = self.original
