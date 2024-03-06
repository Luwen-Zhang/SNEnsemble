cfg = {
    "database": "composite_database_layup_modulus",
    "continuous_feature_names": [
        "Fibre Volume Fraction",
        "Fibre Weight Fraction",
        "Fibre Weight Fraction (0-deg)",
        "Fibre Weight Fraction (45-deg)",
        "Fibre Weight Fraction (90-deg)",
        "Fibre Weight Fraction (Other Dir.)",
        "Ultimate Tensile Stress",
        "Ultimate Compressive Stress",
        "Ultimate Tensile Strain",
        "Ultimate Compressive Strain",
    ],
    "categorical_feature_names": ["Resin Type"],
    "feature_types": {},
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
