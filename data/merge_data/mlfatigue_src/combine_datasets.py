import numpy as np
import pandas as pd

df_names = ["SNL/MSU/DOE", "OptiMat", "Upwind", "FACT"]

dfs = [
    pd.read_excel(
        f"../mlfatigue_output/{s.replace('/', '_')}_fatigue.xlsx", engine="openpyxl"
    )
    for s in df_names
]

variables = [
    # "OptiDat Test Number",
    # "Material_Code",
    # "Sequence",
    "Fiber Volume Fraction",
    "Fiber Weight Fraction",
    "Fiber Weight Fraction (0-deg)",
    "Fiber Weight Fraction (45-deg)",
    "Fiber Weight Fraction (90-deg)",
    "Fiber Weight Fraction (Other Dir.)",
    "Thickness",
    "Tab Thickness",
    "Width",
    "Length",
    "Load Length",
    "Area",
    "Maximum Stress",
    "Minimum Stress",
    "Maximum Strain",
    "Minimum Strain",
    "R-value",
    "Frequency",
    "Ultimate Tensile Stress",
    "Ultimate Compressive Stress",
    "Tensile Modulus",
    "Compressive Modulus",
    "Ultimate Tensile Strain",
    "Ultimate Compressive Strain",
    # "Temperature",
    # "Relative Humidity",
    "Cycles to Failure",
    "log10(Cycles to Failure)",
]

info = pd.DataFrame(index=variables, columns=df_names)

for df_name, df in zip(df_names, dfs):
    # print(df_name)
    desc = {}
    for var in variables:
        # print(var)
        if var in df.columns:
            desc[var] = f"{np.nanmin(df[var]):.2f}~{np.nanmax(df[var]):.2f}"
            info.loc[var, df_name] = desc[var]
        else:
            desc[var] = ""

# print(info)


consistent_order = [
    "Data source",
    "Material_Code",
    "Sequence",
    "Resin Type",
    "Waveform",
    "Control",
] + variables
for df_name, df in zip(df_names, dfs):
    df["Data source"] = df_name
    other_vars = list(np.setdiff1d(df.columns, consistent_order))
    df.columns = [f"{df_name}-{x}" if x in other_vars else x for x in df.columns]

tmp_df_all = pd.concat(dfs, axis=0, ignore_index=True).dropna(
    axis=0, subset=["log10(Cycles to Failure)"]
)

other_vars = list(np.setdiff1d(tmp_df_all.columns, consistent_order))
order = consistent_order + other_vars
df_all = pd.DataFrame(
    columns=consistent_order,
    data=tmp_df_all[consistent_order].values,
    index=np.arange(len(tmp_df_all)),
)
df_all_extended = pd.DataFrame(
    columns=order, data=tmp_df_all[order].values, index=np.arange(len(tmp_df_all))
)
df_all.loc[:, variables] = df_all.loc[:, variables].astype(float)

all_info = df_all[variables].describe()

for var in variables:
    info.loc[var, "Final"] = f"[{df_all[var].min():.2f}, {df_all[var].max():.2f}]"

info.to_csv("../mlfatigue_output/info_03222024.csv")
# df_all.to_excel("../mlfatigue_output/composite_database_02162023.xlsx", index=False, engine="openpyxl")
df_all.to_csv("../mlfatigue_output/composite_database_03222024.csv", index=False)
df_all_extended.to_excel(
    "../mlfatigue_output/composite_database_03222024_with_extended_information.xlsx",
    index=False,
    engine="openpyxl",
)
