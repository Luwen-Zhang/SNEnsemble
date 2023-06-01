cfg = {
    "database": "composite",
    "loss": "mse",
    "bayes_opt": False,
    "n_calls": 50,
    "bayes_epoch": 30,
    "patience": 50,
    "epoch": 200,
    "lr": 0.001,
    "weight_decay": 1e-9,
    "batch_size": 1024,
    "layers": [64, 128, 256, 128, 64],
    "SPACEs": {
        "lr": {
            "type": "Real",
            "low": 1e-4,
            "high": 0.05,
            "prior": "log-uniform",
        },
        "weight_decay": {
            "type": "Real",
            "low": 1e-9,
            "high": 0.05,
            "prior": "log-uniform",
        },
        "batch_size": {
            "type": "Categorical",
            "categories": [32, 64, 128, 256, 512, 1024, 2048],
        },
    },
    "data_splitter": "CycleSplitter",
    "data_imputer": "MissForestImputer",
    "data_processors": [
        ("CategoricalOrdinalEncoder", {}),
        ("NaNFeatureRemover", {}),
        ("VarianceFeatureSelector", {"thres": 1}),
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
