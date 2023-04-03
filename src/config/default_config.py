class DefaultConfig:
    def __init__(self):
        self.cfg = {
            "database": "composite",
            "loss": "mse",
            "bayes_opt": False,
            "bayes_epoch": 30,
            "patience": 50,
            "epoch": 200,
            "lr": 0.003,
            "weight_decay": 0.002,
            "batch_size": 1024,
            "static_params": ["patience", "epoch"],
            "chosen_params": ["lr", "weight_decay", "batch_size"],
            "layers": [64, 128, 256, 128, 64],
            "n_calls": 50,
            "SPACEs": {
                "lr": {
                    "type": "Real",
                    "low": 1e-3,
                    "high": 0.05,
                    "prior": "log-uniform",
                },
                "weight_decay": {
                    "type": "Real",
                    "low": 1e-5,
                    "high": 0.05,
                    "prior": "log-uniform",
                },
                "batch_size": {
                    "type": "Categorical",
                    "categories": [32, 64, 128, 256, 512, 1024, 2048],
                },
            },
            "data_splitter": "MaterialCycleSplitter",
            "data_imputer": "MissForestImputer",
            "data_processors": [
                ("NaNFeatureRemover", {}),
                ("SingleValueFeatureRemover", {}),
                ("UnscaledDataRecorder", {}),
                ("StandardScaler", {}),
            ],
            "data_derivers": [],
            "feature_names_type": {},
            "categorical_feature_names": [
                "Data source",
                "Resin Type",
                "Sequence",
                "Material_Code",
            ],
            "feature_types": ["Fatigue loading", "Material", "Derived"],
            "label_name": ["Cycles to Failure"],
        }
        self.defaults = self.cfg.copy()

    def available_keys(self):
        return list(self.cfg.keys())

    def defaults(self):
        return self.defaults.copy()
