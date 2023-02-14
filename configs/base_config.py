"""
This is a base of all other configuration files. It is not a file for the input of the main script.
"""


class BaseConfig:
    def __init__(self):
        self.data = {
            "database": "FACT_fatigue",
            "loss": "mse",
            "bayes_opt": False,
            "bayes_epoch": 3,
            "patience": 50,
            "epoch": 200,
            "lr": 0.003,
            "weight_decay": 0.002,
            "batch_size": 1024,
            "static_params": ["patience", "epoch"],
            "chosen_params": ["lr", "weight_decay", "batch_size"],
            "layers": [16, 64, 128, 128, 64, 16],
            "n_calls": 100,
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
                    "categories": [32, 64, 128, 256, 512, 1024, 2048, 4096],
                },
            },
            "data_splitter": "MaterialCycleSplitter",
            "data_imputer": "MeanImputer",
            "data_processors": [
                ("IQRRemover", {}),
                # ("MaterialSelector", {"m_code": "MD-DD5P-UP2[0/±45/0]S"}),
                # ("LackDataMaterialRemover", {}),
                # ("FeatureValueSelector", {"feature": "R-value", "value": 0.1}),
                ("NaNFeatureRemover", {}),
                ("SingleValueFeatureRemover", {}),
                ("UnscaledDataRecorder", {}),
                ("StandardScaler", {}),
            ],
            "data_derivers": [],
            "feature_names_type": {},
            "feature_types": ["Fatigue loading", "Material", "Derived"],
            "label_name": ["Cycles to Failure"],
        }
