cfg = {
    "database": "composite_database_10082023_5mat",
    "continuous_feature_names": [
        "Fiber Volume Fraction",
        # "Fiber Weight Fraction",
        # "Fiber Weight Fraction (0-deg)",
        # "Fiber Weight Fraction (45-deg)",
        # "Fiber Weight Fraction (90-deg)",
        # "Fiber Weight Fraction (Other Dir.)",
        # "Thickness",
        # "Tab Thickness",
        # "Width",
        # "Length",
        # "Load Length",
        # "Area",
        "Maximum Stress",
        "Minimum Stress",
        # "Maximum Strain",
        # "Minimum Strain",
        "R-value",
        "Frequency",
        "Ultimate Tensile Stress",
        "Ultimate Compressive Stress",
        # "Tensile Modulus",
        # "Compressive Modulus",
        # "Ultimate Tensile Strain",
        # "Ultimate Compressive Strain",
    ],
    "categorical_feature_names": [],
    "feature_types": {
        "Resin Type": "Material/Specimen",
        "Fiber Volume Fraction": "Material/Specimen",
        "Fiber Weight Fraction": "Material/Specimen",
        "Fiber Weight Fraction (0-deg)": "Material/Specimen",
        "Fiber Weight Fraction (45-deg)": "Material/Specimen",
        "Fiber Weight Fraction (90-deg)": "Material/Specimen",
        "Fiber Weight Fraction (Other Dir.)": "Material/Specimen",
        "Thickness": "Material/Specimen",
        "Tab Thickness": "Material/Specimen",
        "Width": "Material/Specimen",
        "Length": "Material/Specimen",
        "Load Length": "Material/Specimen",
        "Area": "Material/Specimen",
        "Maximum Stress": "Fatigue loading",
        "Minimum Stress": "Fatigue loading",
        "Maximum Strain": "Fatigue loading",
        "Minimum Strain": "Fatigue loading",
        "R-value": "Fatigue loading",
        "Frequency": "Fatigue loading",
        "Ultimate Tensile Stress": "Material/Specimen",
        "Ultimate Compressive Stress": "Material/Specimen",
        "Tensile Modulus": "Material/Specimen",
        "Compressive Modulus": "Material/Specimen",
        "Ultimate Tensile Strain": "Material/Specimen",
        "Ultimate Compressive Strain": "Material/Specimen",
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
            "SuppStressDeriver",
            {
                "derived_name": "Support Stress",
                "stacked": True,
                "max_stress_col": "Maximum Stress",
                "min_stress_col": "Minimum Stress",
                "ucs_col": "Ultimate Compressive Stress",
                "uts_col": "Ultimate Tensile Stress",
                "relative": True,
                "absolute": False,
            },
        ],
    ],
    "data_processors": [
        ["CategoricalOrdinalEncoder", {}],
        ["NaNFeatureRemover", {}],
        ["VarianceFeatureSelector", {"thres": 1}],
        ["StandardScaler", {}],
    ],
    "label_name": ["log10(Cycles to Failure)"],
}
