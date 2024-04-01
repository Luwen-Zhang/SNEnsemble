cfg = {
    "database": "composite_database_layup_modulus_03302024",
    "continuous_feature_names": [
        "Fiber Volume Fraction",
        "Fiber Weight Fraction",
        "Fiber Weight Fraction (0-deg)",
        "Fiber Weight Fraction (45-deg)",
        "Fiber Weight Fraction (90-deg)",
        "Fiber Weight Fraction (Other Dir.)",
        "Ultimate Tensile Stress",
        "Ultimate Compressive Stress",
        "Ultimate Tensile Strain",
        "Ultimate Compressive Strain",
    ],
    "categorical_feature_names": ["Resin Type"],
    "feature_types": {},
    "layers": [512, 256, 128, 64],
    "data_derivers": [
        [
            "LayUpSequenceDeriver",
            {
                "sequence_column": "Sequence",
                "derived_name": "Lay-up Sequence",
                "stacked": False,
            },
        ],
        [
            "NumLayersDeriver",
            {
                "sequence_column": "Sequence",
                "derived_name": "Number of Layers",
                "stacked": False,
            },
        ],
    ],
    "data_processors": [
        ["CategoricalOrdinalEncoder", {}],
        ["NaNFeatureRemover", {}],
        ["VarianceFeatureSelector", {"thres": 1}],
        ["StandardScaler", {}],
    ],
    "label_name": ["Tensile Modulus"],
}
