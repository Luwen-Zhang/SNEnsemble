from utils import *

data_path = "../mlfatigue_data/SNL_MSU_DOE_raw.xlsx"
exclude_sheet = ["Resins", "Fabrics", "Environmental", "Recent misc."]

dfs = pd.read_excel(data_path, engine="openpyxl", sheet_name=None)

sheet_names = list(dfs.keys())
for ex in exclude_sheet:
    if ex in sheet_names:
        sheet_names.pop(sheet_names.index(ex))

df_all = pd.concat([dfs[name] for name in sheet_names], axis=0, ignore_index=True)

name_database = "SNL_MSU_DOE"

# ------
modifications = [
    ("Testing Temperature, OC", remove_s, dict(s=" ̊C")),
    ("%, 45 Deg", remove_s, dict(s=" G")),
    ("%, 0 Deg", remove_s, dict(s=" C")),
    ("other %", remove_s, dict(s=" G")),
    ("Vf, %", cal_fraction, dict(s="/")),
    ("%, 0 Deg", cal_fraction, dict(s="-")),
    ("%, 45 Deg", cal_fraction, dict(s="-")),
    ("Thickness, mm", conditional_remove, dict(s="mm dia")),
    ("Thickness, mm", conditional_remove, dict(s="/")),
    ("Max. Stress, MPa", remove_s, dict(s="*")),
    ("Max. Stress, MPa", remove_s, dict(s="+")),
    ("Max. Stress, MPa", conditional_remove, dict(s="Newtons")),
    ("Min. Stress, MPa", conditional_remove, dict(s="v")),
    ("R-value", conditional_remove, dict(s="*")),
    ("R-value", conditional_replace, dict(s1="static compression", s2="static")),
    ("Max. % Strain", conditional_remove, dict(s="----")),
    ("Max. % Strain", remove_s, dict(s="+")),
    ("Max. % Strain", remove_strs, dict()),
    ("Min. % Strain", conditional_remove, dict(s="Runout")),
    ("E, GPa (0.1-0.3%)", conditional_remove, dict(s="----")),
    ("E, GPa", remove_strs, dict()),
    ("Initial cracking strain, %", remove_strs, dict()),
    ("Runout", str2num, dict(s="Runout", n=1)),
    ("Runout", str2num, dict(s="runout", n=1)),
    ("Resin Type", conditional_replace_strs, dict(s1="Epoxy", s2="EP")),
    ("Resin Type", conditional_replace_strs, dict(s1="EP", s2="EP")),
    ("Resin Type", conditional_replace_strs, dict(s1="TP", s2="TP")),
    ("Resin Type", conditional_replace_strs, dict(s1="UP", s2="PE")),
    ("Resin Type", conditional_replace_strs, dict(s1="VE", s2="VE")),
    ("Resin Type", replace_strs, dict(s1="pDCPD", s2=np.nan)),
    ("Resin Type", replace_strs, dict(s1="PU", s2=np.nan)),
    ("Resin Type", replace_strs, dict(s1="T", s2=np.nan)),
    ("Freq., Hz", conditional_remove, dict(s="mm/s")),
]
# ------
unknown_categorical = ["Control", "Waveform"]
# ------
na_to_0 = [
    "%, 0 Deg",
    "%, 45 Deg",
    "%, 90 Deg",
    "other %",
    "45 Deg fabric",
    "90 deg fabric",
    "Runout",
]
# ------
merge = [
    ("E, GPa (0.1-0.3%)", "E, GPa"),
    ("Freq., Hz or mm/s", "Freq., Hz"),
]
# ------
adds = []
# ------
force_remove_idx = []
# ------
static_crit = lambda df: df["Cycles"] == 1
fatigue_crit = None
# ------
init_r_col = "R-value"
init_max_stress_col = "Max. Stress, MPa"
init_min_stress_col = "Min. Stress, MPa"
init_max_strain_col = None
init_min_strain_col = None
# ------
fill_absent_stress_r = True
# ------
eval_min_stress = init_min_stress_col
# ------
extract_separate_e = [
    {
        "init_col": "E, GPa",
        "comp_crit": lambda row: np.isnan(row["Max. Stress, MPa"]),
        "comp_name": "Eic",
        "tensile_name": "Eit",
    }
]
# ------
mat_code = lambda df: [
    str(df.loc[x, "Material"]) + str(df.loc[x, "Lay-up"]) for x in range(len(df))
]
# ------
init_seq_col = "Lay-up"
# ------
get_static = True
static_to_extract = [
    init_max_stress_col,
    init_min_stress_col,
    "Max. % Strain",
    "Min. % Strain",
    "Eit",
    "Eic",
]
static_prefix = "(Static) "
static_merge_exist = None
# ------
illegal_crit = {
    static_prefix
    + init_max_stress_col: lambda df: df[static_prefix + "Max. Stress, MPa"]
    < 0,
    static_prefix
    + init_min_stress_col: lambda df: df[static_prefix + "Min. Stress, MPa"]
    > 0,
    static_prefix + "Max. % Strain": lambda df: df[static_prefix + "Max. % Strain"] < 0,
    static_prefix + "Min. % Strain": lambda df: df[static_prefix + "Min. % Strain"] > 0,
}
# ------
not_runout_crit = lambda df: df["Runout"] == 0
# ------
init_cycle_col = "Cycles"
# ------
special_treatment = None

name_mapping = {
    # 'Material': 'Material',
    # 'Resin Type': 'Resin Type',
    "Vf, %": "Fiber Volume Fraction",
    "%, 0 Deg": "Fiber Weight Fraction (0-deg)",
    "%, 45 Deg": "Fiber Weight Fraction (45-deg)",
    "%, 90 Deg": "Fiber Weight Fraction (90-deg)",
    "other %": "Fiber Weight Fraction (Other Dir.)",
    "Thickness, mm": "Thickness",
    "Max. Stress, MPa": "Maximum Stress",
    "Min. Stress, MPa": "Minimum Stress",
    "R-value": "R-value",
    "Freq., Hz": "Frequency",
    # 'E, GPa': 'Initial Elastic Modulus',
    "Max. % Strain": "Maximum Strain",
    "Min. % Strain": "Minimum Strain",
    "Cycles": "Cycles to Failure",
    # 'Moisture Gain, %': 'Moisture Gain',
    # 'Testing Temperature, OC': 'Temperature',
    "Width, mm": "Width",
    static_prefix + "Max. Stress, MPa": "Ultimate Tensile Stress",
    static_prefix + "Min. Stress, MPa": "Ultimate Compressive Stress",
    static_prefix + "Eit": "Tensile Modulus",
    static_prefix + "Eic": "Compressive Modulus",
    static_prefix + "Max. % Strain": "Ultimate Tensile Strain",
    static_prefix + "Min. % Strain": "Ultimate Compressive Strain",
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
    special_treatments=special_treatment,
)
