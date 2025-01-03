from utils import *

data_path = "../mlfatigue_data/Upwind_combine.xlsx"

df_all = pd.read_excel(data_path, engine="openpyxl", sheet_name=None)["Sheet1"]

df_all["Resin Type"] = "EP"
df_all = df_all.dropna(axis=0, subset=["Plate-Resin System-Resin"]).reset_index(
    drop=True
)

name_database = "Upwind"

# ------
modifications = [
    ("widthmiddle", cal_fraction, dict(s=" / ")),
    ("widthmiddle", cal_fraction, dict(s="/")),
    ("T02 max[ºC]", remove_strs, dict()),
    ("Plate-Fibre Weight Percentage", remove_strs, dict()),
    ("Plate-Glass Density", remove_strs, dict()),
    ("Plate-Laminate-Width", remove_strs, dict()),
    ("Plate-Fibre Weight", remove_strs, dict()),
    ("Plate-Laminate-Thickness Adjustment Mould", remove_strs, dict()),
    ("Plate-Laminate-Length", remove_strs, dict()),
    ("Plate-Fibre Mass", remove_strs, dict()),
]
# ------
unknown_categorical = ["Control", "Waveform"]
# ------
na_to_0 = []
# ------
merge = []
# ------
adds = []
# ------
force_remove_idx = np.where(df_all["Environment"] != "Ambient")[0]
# ------
static_crit = lambda df: (df["test type"] == "STT") | (df["test type"] == "STC")
fatigue_crit = lambda df: df["test type"] == "CA"
# ------
init_r_col = "R-value"
init_max_stress_col = "σmax[MPa]"
init_min_stress_col = None
init_max_strain_col = None
init_min_strain_col = None
# ------
fill_absent_stress_r = False
# ------
eval_min_stress = "Minimum Stress"
# ------
extract_separate_e = [
    {
        "init_col": "σmax[MPa]",
        "comp_crit": lambda row: row["σmax[MPa]"] < 0,
        "comp_name": "Maximum Compressive Stress",
        "tensile_name": "Maximum Tensile Stress",
    },
    {
        "init_col": "E_avg[GPa]",
        "comp_crit": lambda row: row["σmax[MPa]"] < 0,
        "comp_name": "Eic",
        "tensile_name": "Eit",
    },
]
# ------
mat_code = lambda df: [
    str(df.loc[x, "Property Plate"]) + str(df.loc[x, "geometry"])
    for x in range(len(df))
]
# ------
init_seq_col = "Plate-Lay-up"
# ------
get_static = True
static_to_extract = [
    "Maximum Tensile Stress",
    "Maximum Compressive Stress",
    "Eit",
    "Eic",
]
static_prefix = "(Static) "
static_merge_exist = None
# ------
illegal_crit = {}
# ------
not_runout_crit = lambda df: df["runout"] != "y"
# ------
init_cycle_col = "cycles to failurefatigue"


# ------
def fill_all_notna_E(df_fatigue):
    where_E_avg_notna = np.where(pd.notna(df_fatigue["E_avg[GPa]"]))[0]
    df_fatigue.loc[where_E_avg_notna, "(Static) Eit"] = df_fatigue.loc[
        where_E_avg_notna, "E_avg[GPa]"
    ]
    df_fatigue.loc[where_E_avg_notna, "(Static) Eic"] = df_fatigue.loc[
        where_E_avg_notna, "E_avg[GPa]"
    ]
    return df_fatigue


special_treatments = [fill_all_notna_E]

name_mapping = {
    # 'Table', ##### Not useful
    "OptiDAT nr.": "OptiDat Test Number",
    # 'identification', ##### Not useful
    # 'Property Plate', ##### Used to generate Material_Code
    # 'laboratory', ##### Not useful
    # 'angle': 'Load Angle',
    # 'geometry', ##### Used to generate Material_Code
    # 'material', ##### Not useful
    "average thicknessmiddle": "Thickness",
    "widthmiddle": "Width",
    "area[mm2]": "Area",
    "length total[mm]": "Length",
    "length gaugeaverage": "Load Length",
    # 'workpackageI', ##### Not useful
    # 'start date', ##### Not useful
    # 'end date', ##### Not useful
    # 'test type', ##### Identify type
    "R-value": "R-value",
    # 'load (max)[kN]': 'Maximum Load', ##### Using Maximum Stress directly
    # 'average_emax': 'Average Strain',
    # 'poissonaverage': 'Average Poisson Ratio',
    "σmax[MPa]": "Maximum Stress",
    # 'G[Gpa]': 'Shear Modulus',
    "cycles to failurefatigue": "Cycles to Failure",
    # 'runout', ##### Not useful
    # 'loading rate[mm/min]': 'Displacement Rate',
    "loading rate[Hz]": "Frequency",
    # 'E_avg[GPa]': 'Modulus (Tensile or Compressive)', ##### Fill Static Elastic/Compressive Modulus.
    # 'machine', ##### Not useful
    # 'control mode', ##### Not useful
    # 'special fixture', ##### Not useful
    # 'tabs remarks', ##### Not useful
    "T02 max[ºC]": "Temperature",
    # 'Environment', ##### Most of them are "Ambient"
    # 'reference document(s)', ##### Not useful
    # 'remarks', ##### Not useful
    # 'invalid', ##### Not useful
    # 'strain gaugesmiddle', ##### Not useful
    # 'grip pressure[MPa]', ##### How to use it?
    # 'number of layers': 'Number of Layers',
    "tab thickness": "Tab Thickness",
    # 'Plate-Fibre Type',
    # 'Plate-Mould', ##### Not useful
    # 'Plate-Date of Arrival', ##### Not useful
    # 'Plate-Glass Density', ##### To calculate Volume Fraction
    # 'Plate-Laminate-Width': 'Laminate Width', ##### Use Width directly.
    # 'Plate-Fibre Weight': 'Fibre Weight', ##### Use Fibre Weight Fraction
    # 'Plate-Project', ##### Not useful
    # 'Plate-Laminate-Thickness Adjustment Mould': 'Thickness Adjustment Mould', ##### not useful
    # 'Plate-Injection Date', ##### Not useful
    # 'Plate-Resin System-Mix Ratio', ##### Most of them are the same
    # 'Plate-Material', ##### Not useful
    # 'Plate-Weaver', ##### Not useful
    # 'Plate-Laminate-Length': 'Laminate Length', misleading
    # 'Plate-Remarks', ##### Not useful
    # 'Plate-Resin Density(liquid)', ##### To calculate Volume Fraction
    # 'Plate-Fibre Volume Fraction', ##### Incomplete, recalculate
    # 'Plate-Vacuum-Curing', ##### Most of them are the same
    # 'Plate-Layers', ##### Replicated
    # 'Plate-article nr.', ##### Not useful
    # 'Plate-Prepared Resin Mixture-Resin': 'Resin Mixture Resin Weight', ##### Not useful
    # 'Plate-Post-cure', ##### How to use it?
    # 'Plate-Resin System-Resin', ##### Not useful
    # 'Plate-Vacuum-Injection': 'Vacuum Injection', ##### Not useful
    # 'Plate-Prepared Resin Mixture-Hardener': 'Resin Mixture Hardener Weight', ##### Not useful
    # 'Plate-Lay-up': 'Lay-up',
    # 'Plate-Curing Tabs Glue ', ##### How to use it?
    # 'Plate-Fibre Mass', ##### To calculate Volume Fraction
    "Plate-Fibre Weight Percentage": "Fiber Weight Fraction",
    # 'Plate-Delta Cp', ##### Not useful
    # 'Plate-Fibre Weight Fraction', ##### Too many absence
    # 'Plate-Tg', ##### Too many absence
    # 'Plate-Prepared Resin Mixture-Mix': 'Resin Mixture Weight', ##### Not useful
    # 'Plate-Void Content', ##### Too many absence
    # 'Plate-Prepared Resin Mixture-Ratio': 'Resin Mixture Hardener Fraction', ##### Not useful
    # 'Plate-Resin System-Hardener', ##### Not useful
    # 'Plate-Material Density', ##### Too many absence
    # 'Plate-Resin Density(cured)', ##### Data seems not compatible with Resin Density(liquid)
    # 'Plate-Operater' ##### Not useful
    ################### Added features ###################
    # 'Maximum Tensile Stress': 'Maximum Tensile Stress',
    # 'Maximum Compressive Stress': 'Maximum Compressive Stress',
    "(Static) Maximum Tensile Stress": "Ultimate Tensile Stress",
    "(Static) Maximum Compressive Stress": "Ultimate Compressive Stress",
    "Minimum Stress": "Minimum Stress",
    "(Static) Eit": "Tensile Modulus",
    "(Static) Eic": "Compressive Modulus",
}

main_process(
    df_all=df_all,
    name_database=name_database,
    name_mapping=name_mapping,
    modifications=modifications,
    unknown_categorical=unknown_categorical,
    na_to_0=na_to_0,
    merge=merge,
    adds=adds,
    force_remove_idx=force_remove_idx,
    init_r_col=init_r_col,
    init_max_stress_col=init_max_stress_col,
    init_min_stress_col=init_min_stress_col,
    init_max_strain_col=init_max_strain_col,
    init_min_strain_col=init_min_strain_col,
    init_seq_col=init_seq_col,
    init_cycle_col=init_cycle_col,
    mat_code=mat_code,
    static_to_extract=static_to_extract,
    static_prefix=static_prefix,
    static_crit=static_crit,
    static_merge_exist=static_merge_exist,
    fatigue_crit=fatigue_crit,
    not_runout_crit=not_runout_crit,
    illegal_crit=illegal_crit,
    fill_absent_stress_r=fill_absent_stress_r,
    eval_min_stress=eval_min_stress,
    extract_separate_e=extract_separate_e,
    get_static=get_static,
    special_treatments=special_treatments,
)
