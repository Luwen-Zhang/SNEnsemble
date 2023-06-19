cfg = {
    "database": "composite_database_04212023_1_mat",
    "feature_names_type": {
        "Resin Type": 2,
        "Fibre Volume Fraction": 1,
        "Fibre Weight Fraction": 1,
        "Percentage of Fibre in 0-deg Direction": 1,
        "Percentage of Fibre in 45-deg Direction": 1,
        "Percentage of Fibre in 90-deg Direction": 1,
        "Percentage of Fibre in Other Direction": 1,
        "Thickness": 1,
        "Tab Thickness": 1,
        "Width": 1,
        "Maximum Width": 1,
        "Minimum Width": 1,
        "Length": 1,
        "Gauge Length": 1,
        "Load Length": 1,
        "Area": 1,
        "Maximum Stress": 0,
        "Minimum Stress": 0,
        "Maximum Strain": 0,
        "Minimum Strain": 0,
        "R-value": 0,
        "Frequency": 0,
        "Static Maximum Tensile Stress": 1,
        "Static Maximum Compressive Stress": 1,
        "Static Elastic Modulus": 1,
        "Static Compressive Modulus": 1,
        "Static Maximum Tensile Strain": 1,
        "Static Maximum Compressive Strain": 1,
    },
    "data_derivers": [
        [
            "MinStressDeriver",
            {
                "derived_name": "Minimum Stress",
                "stacked": True,
                "intermediate": True,
                "max_stress_col": "Maximum Stress",
                "r_value_col": "R-value",
            },
        ],
        [
            "WalkerStressDeriver",
            {
                "derived_name": "Walker Eq Stress",
                "stacked": True,
                "max_stress_col": "Maximum Stress",
                "r_value_col": "R-value",
                "power_index": 0.5,
            },
        ],
        [
            "SuppStressDeriver",
            {
                "derived_name": "Support Stress",
                "stacked": True,
                "max_stress_col": "Maximum Stress",
                "min_stress_col": "Minimum Stress",
                "ucs_col": "Static Maximum Compressive Stress",
                "uts_col": "Static Maximum Tensile Stress",
                "relative": True,
            },
        ],
        [
            "DegLayerDeriver",
            {
                "sequence_column": "Sequence",
                "derived_name": "deg_layers",
                "col_names": [
                    "0-deg layers",
                    "45-deg layers",
                    "90-deg layers",
                    "Other-deg layers",
                ],
                "stacked": True,
            },
        ],
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
        ["SampleWeightDeriver", {"derived_name": "Sample Weight", "stacked": False}],
    ],
    "data_processors": [
        ["CategoricalOrdinalEncoder", {}],
        ["NaNFeatureRemover", {}],
        ["VarianceFeatureSelector", {"thres": 1}],
        ["StandardScaler", {}],
    ],
    "feature_types": ["Fatigue loading", "Material", "Categorical", "Derived"],
    "label_name": ["log(Cycles to Failure)"],
}
