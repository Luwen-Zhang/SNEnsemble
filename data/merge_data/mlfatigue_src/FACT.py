from utils import *

data_path = "../mlfatigue_data/FACT_raw.xlsx"

df_all = pd.read_excel(data_path, engine="openpyxl", sheet_name=None)["Sheet1"]

df_all.drop(0, axis=0, inplace=True)
df_all.reset_index(drop=True, inplace=True)

df_all["Resin Type"] = df_all["material "].copy()

name_database = "FACT"

# ------
modifications = [
    (0, fill_na, dict(n=0)),
    ("+/-45", fill_na, dict(n=0)),
    (90, fill_na, dict(n=0)),
    ("other direction", fill_na, dict(n=0)),
    ("Resin Type", replace_strs, dict(s1="GE", s2="EP")),
    ("Resin Type", replace_strs, dict(s1="GP", s2="PE")),
    ("Resin Type", replace_strs, dict(s1="GV", s2="VE")),
    ("wave", replace_strs, dict(s1="s", s2="Sinusoidal")),
    ("wave", replace_strs, dict(s1="t", s2="Triangular")),
    ("control", replace_strs, dict(s1="d", s2="Displacement")),
    ("control", replace_strs, dict(s1="l", s2="Load")),
    ("UCS", col_abs, dict()),
    ("UCS", col_neg, dict()),
    ("e_UCS", col_abs, dict()),
    ("e_UCS", col_neg, dict()),
]
# ------
unknown_categorical = []
# ------
na_to_0 = []
# ------
merge = []
# ------
adds = []
# ------
not_humidity = np.intersect1d(
    np.where(df_all["environment"] != "s")[0], np.where(df_all["environment"] != "w")[0]
)
not_temp = np.where(df_all["Temperature control?"] != "y")[0]
retain = np.intersect1d(not_humidity, not_temp)
force_remove_idx = np.setdiff1d(np.array(df_all.index), retain)
# ------
static_crit = lambda df: df["test type"] != "CA"
fatigue_crit = None
# ------
init_r_col = "R-value"
init_max_stress_col = "s_max"
init_min_stress_col = None
init_max_strain_col = "e_max"
init_min_strain_col = None
# ------
fill_absent_stress_r = False
# ------
eval_min_stress = "Minimum Stress"
# ------
extract_separate_e = None
# ------
mat_code = lambda df: [
    str(df.loc[x, "material "]) + str(df.loc[x, "laminate"]) for x in range(len(df))
]
# ------
init_seq_col = "laminate"
# ------
get_static = False
static_to_extract = []
static_prefix = None
static_merge_exist = None
# ------
illegal_crit = {}
# ------
not_runout_crit = lambda df: df["runout?"] != "y"
# ------
init_cycle_col = "No. of cycles to failure"
# ------
special_treatments = None

name_mapping = {
    "optiDAT nr.": "OptiDat Test Number",
    # 'Optimat/FACT name', ##### Not useful
    # 'data delivered under name', ##### Not useful
    # 'from plate',##### empty
    # 'tested at laboratory', ##### Not useful
    # 'laminate': 'Lay-up',
    # 'cut angle': 'Cut Angle', ##### empty
    # 'chop', ##### What is it?
    0: "Layers of Fiber in 0-deg Direction",
    "+/-45": "Layers of Fiber in 45-deg Direction",
    90: "Layers of Fiber in 90-deg Direction",
    "other direction": "Layers of Fiber in Other Direction",
    # 'Resin', ##### Not useful
    # 'Fibre', ##### Not useful
    "FVF": "Fiber Volume Fraction",
    # 'porosity': 'Porosity', ##### Too much absence
    # 'Barcol hardness': 'Barcol Hardness',
    # 'Manufactured by/Quality Assurance Specification', ##### Not useful
    # 'production method',##### Not useful
    # 'Geometry', ##### Not useful
    # 'material ': 'Resin Type', ##### Not useful
    "thickness (old)": "Thickness",
    # 'average thickness (new)', ##### empty
    "maximum width": "Width",
    # "minimum width": "Minimum Width", #### Incomplete compared to maximum width
    # 'average width (new)', ##### empty
    "area": "Area",
    "length": "Length",
    "load length": "Load Length",
    # 'radius of waist': 'Radius of Waist',
    # 'specimen standard', ##### empty
    # 'prog', ##### Not useful
    # 'start test', ##### Not useful
    # 'end of test', ##### Not useful
    # 'test type', ##### As filter
    "R-value": "R-value",
    # 'F_max',##### empty
    # 'Fmax during fatigue', ##### empty
    # 'Fmax during static test', ##### empty
    # 'deviation of Fmax>2% w.r.t. test frame settings during fatigue?', ##### empty
    # 'F_max 90° direction', ##### empty
    "e_max": "Maximum Strain",
    # 'e_max 90°', ##### empty
    # 'e_max 45°', ##### empty
    "s_max": "Maximum Stress",
    # 'normalized s_max', ##### Not useful
    # 'shear strain (e12)', ##### empty
    # 'shear strength', ##### empty
    # 'Shear Modulus (G12)', ##### empty
    "No. of cycles to failure": "Cycles to Failure",
    # 'level', ##### empty
    # 'Unnamed: 51', ##### empty
    # 'no. of cycles', ##### empty
    # 'Unnamed: 53', ##### empty
    # 'No. of spectrum passes to failure', ##### To much absence, how to use it?
    # 'failure mode', ##### To much absence
    # 'runout?',
    # 'R-value.1', ##### empty
    # 'F_max.1', ##### empty
    # 'e_max.1', ##### empty
    # 's_max.1', ##### empty
    # 'No. of cycles in block 2', ##### empty
    "UTS": "Ultimate Tensile Stress",
    "UCS": "Ultimate Compressive Stress",
    "e_UTS": "Ultimate Tensile Strain",
    "e_UCS": "Ultimate Compressive Strain",
    # 'UTS (mat. spec.)', ##### What is it?
    # 'UCS (mat. spec.)', ##### What is it?
    # 'corr. UTS', ##### What is it?
    # 'corr. UCS', ##### What is it?
    # 'RTS', ##### empty
    # 'RCS', ##### empty
    # 'e_RTS', ##### empty
    # 'e_RCS', ##### empty
    # 'LRU': 'Stress Rate',
    # 'SRU': 'Strain Rate',
    # 'SRU.1', ##### empty
    # 'SRU speed deviation ? 2% during static test?', ##### empty
    # 'LRF', ##### What is it?
    # 'SRF', ##### What is it?
    # 'SRF.1', ##### What is it?
    "wave": "Waveform",  ##### Not useful
    "fconstant": "Frequency",
    # 'test time', ##### Not useful
    # 'fconstant block 2', ##### empty
    "Eit": "Tensile Modulus",
    "Eic": "Compressive Modulus",
    # 'Eft',
    # 'Efc',
    # 'ILSS', ##### empty
    # 's_flex', ##### empty
    # 'test machine', ##### empty
    "control": "Control",  ##### All load control
    # 'grip', ##### Not useful
    # 'ABG', ##### What is it?
    "Temperature": "Temperature",
    # 'Temperature control?',
    # 'Preconditioned?', ##### empty
    # 'environment', ##### All "d", What is it?
    "RH": "Relative Humidity",
    # 'Test condition', ##### empty
    # 'ref.', ##### Not useful
    # 'NB', ##### What is it?
    # 'RB', ##### What is it?
    # 's_res', ##### empty
    # 'N10sr', ##### To much absence
    # 'specimen shape', ##### Not useful
    # 'Remarks', ##### empty
    # 'Bending', ##### empty
    # 'buckling', ##### empty
    # 'temperature failure or temperature above 35 °C', ##### empty
    # 'tab failure', ##### empty
    # 'delaminated', ##### empty
    # 'incomplete measurement data available',  ##### empty
    # 'Strain calculated using E', ##### empty
    # 'Premature failure in RST', ##### empty
    # "Poissons's ratio", ##### empty
    # 'Strain measurement equipment', ##### empty
    # 'Strain measurement equipment (2)', ##### empty
    # 'Grip pressure' ##### empty
    "Minimum Stress": "Minimum Stress",
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
