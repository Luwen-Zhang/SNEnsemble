from utils import *

data_path = "../mlfatigue_data/OptiMat_combine.xlsx"

df_all = pd.read_excel(data_path, engine="openpyxl", sheet_name=None)["Sheet1"]

df_all["Resin Type"] = "EP"

name_database = "OptiMat"

# ------
modifications = [
    ("Ncycles", remove_strs, dict()),
    ("Lnominal", remove_strs, dict()),
    ("R-value1", remove_strs, dict()),
    ("ratedisplacement", remove_strs, dict()),
    ("taverage", remove_s, dict(s="' ")),
    ("taverage", remove_strs, dict()),
    ("Temp.", remove_strs, dict()),
    ("Geometry-tab thickness(for UD2; for MD2)", remove_strs, dict()),
    ("Geometry-t(for UD2; for MD2)", remove_strs, dict()),
    ("control", replace_strs, dict(s1="l, d, s", s2="UNK")),
    ("control", replace_strs, dict(s1="l, d", s2="UNK")),
    # l, d should be for residual strength tests, but is recorded for some CA tests.
    ("control", replace_strs, dict(s1="l ", s2="Load")),
    ("control", replace_strs, dict(s1="l", s2="Load")),
    ("control", replace_strs, dict(s1="d", s2="Displacement")),
    ("control", replace_strs, dict(s1="l", s2="Load")),
    ("control", replace_strs, dict(s1="d", s2="Displacement")),
    ("epsstatic", col_abs, dict()),
]
# ------
unknown_categorical = ["Waveform"]
# ------
na_to_0 = ["Laminate- 0°", "Laminate- 45°", "Laminate- -45°", "Laminate- 90°"]
# ------
merge = []
# ------
adds = [("Laminate- -45°", "Laminate- 45°")]
# ------
nan_cycle = np.where(pd.isna(df_all.loc[:, "Ncycles"]))[0]
zero_cycle = np.where(df_all.loc[:, "Ncycles"] == 0)[0]
invalid_cycle = np.union1d(
    np.where(df_all.loc[:, "Ncycles"] == 0.5)[0],
    np.where(df_all.loc[:, "Ncycles"] == 1.5)[0],
)
str_freq = np.where([isinstance(x, str) for x in df_all["f"]])[0]
# Remove tests with temperature control or humidity control
env_control = np.where(df_all["Environment"] != "d, RT")[0]
force_remove_idx = np.union1d(
    np.union1d(np.union1d(np.union1d(nan_cycle, zero_cycle), invalid_cycle), str_freq),
    env_control,
)
# ------
static_crit = lambda df: df["Ncycles"] == 1
fatigue_crit = lambda df: df["Test type"] == "CA"
# ------
init_r_col = "R-value1"
init_max_stress_col = "smax"
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
        "init_col": "smax,static",
        "comp_crit": lambda row: row["smax,static"] < 0,
        "comp_name": "(Static) Maximum Compressive Stress",
        "tensile_name": "(Static) Maximum Tensile Stress",
    },
    {
        "init_col": "epsstatic",
        "comp_crit": lambda row: row["smax,static"] < 0,
        "comp_name": "(Static) Maximum Compressive Strain",
        "tensile_name": "(Static) Maximum Tensile Strain",
    },
]

# ------
mat_code = lambda df: [
    str(df.loc[x, "Property Plate"])
    + str(df.loc[x, "Property Geometry"])
    + str(df.loc[x, "Property Laminate"])
    for x in range(len(df))
]
# ------
init_seq_col = "Laminate-lay-up"
# ------
get_static = True
static_to_extract = [
    "(Static) Maximum Tensile Stress",
    "(Static) Maximum Compressive Stress",
    "Eit",
    "Eic",
    "(Static) Maximum Tensile Strain",
    "(Static) Maximum Compressive Strain",
]
static_prefix = "(Static) "
static_merge_exist = ["Eit", "Eic"]
# ------
illegal_crit = {}
# ------
not_runout_crit = lambda df: df["runout"] != "y"
# ------
init_cycle_col = "Ncycles"


# ------
def neg_static_ucs(df_fatigue):
    df_fatigue["(Static) Maximum Compressive Strain"] *= -1
    return df_fatigue


special_treatments = [neg_static_ucs]

name_mapping = {
    # 'Table', ###### Not useful
    "optiDAT nr.": "OptiDat Test Number",
    # 'Name', ###### Not useful
    # 'Property Plate': 'Plate', ###### Used
    "Fibre Volume Fraction": "Fiber Volume Fraction",
    # 'Lab', ###### Not useful
    # 'Property Laminate': 'Laminate', ###### Used
    # 'Cut angle': 'Laminate Cut Angle', ##### Not useful
    # 'Manufacturer', ###### Not useful
    # 'Property Geometry': 'Geometry', ###### Used
    # 'Material', ###### Not useful
    "taverage": "Thickness",
    # "wmax": "Maximum Width",
    # "wmin": "Minimum Width",
    "waverage": "Width",
    "area": "Area",
    # 'Lnominal': 'Length(nominal)', ##### Not useful
    # 'Lmeasured': 'Length(measured)', ##### Not useful
    # "Lgauge": "Gauge Length", ##### same as load length but less complete
    # 'Radius (waist)', ###### Not useful
    # 'TG and phase', ###### Not useful
    # 'Phase', ###### Classification
    # 'start date',##### Not useful
    # 'end date', ##### Not useful
    # 'Test type', ###### Classification
    "R-value1": "R-value",
    # 'Fmax': 'Maximum Load', ##### Not useful, use stress directly.
    # 'Ferror', ##### Not necessary
    # 'Fstatic': 'Static Maximum Load',  ##### Not useful
    # 'Ffatigue', ###### Looks similar with Fmax
    # 'Fmax, 90°': 'Transverse Maximum Load', ##### Not useful
    "epsmax": "Maximum Strain",
    # 'epsstatic': 'Static Strain',
    # 'epsfatigue', ###### same as epsmax
    # 'eps90°': 'Transverse Maximum Strain',
    # 'eps45°': '45-deg Maximum Strain',
    # 'Nu': 'Poisson\'s Ratio',
    "smax": "Maximum Stress",
    # 'smax,static': 'Static Maximum Stress',
    # 'smax, fatigue':, ###### same as smax
    # 'shear strain': 'Shear Strain',
    # 'shear strength': 'Shear Strength',
    # 'shear modulus (near other moduli)': 'Shear Modulus',
    "Torquemax": "Maximum Torque",
    # 'Torquefatigue':, ###### Too many absense
    # 'Torquestatic', ##### Too many absense
    "Ncycles": "Cycles to Failure",
    # 'level',  ##### Not useful?
    # 'levelcalc', ##### Not useful?
    # 'Nspectrum', ##### Too many absense, Classification?
    # 'failure mode',  ##### Not useful?
    # 'runout',  ##### Not useful
    # 'R-value2', ###### How to use block 2?
    # 'Fmax2', ###### How to use block 2?
    # 'epsmax2', ###### How to use block 2?
    # 'smax2', ###### How to use block 2?
    # 'Ncycles2', ###### How to use block 2?
    # 'ratedisplacement': 'Strain Rate',
    "f": "Frequency",
    # 'f2', ###### How to use block 2?
    # "Eit": "Elastic Modulus",
    # "Eic": "Compressive Modulus",
    # 'Eft': 'Final Tensile Modulus', ##### Too many absense
    # 'Efc': 'Final Compressive Modulus', ##### Too many absense
    # 'Elur,front', ##### Too many absense
    # 'Elur, back', ##### Too many absense
    # 'eps_max_t': 'Maximum Strain (to Elastic Modulus)', ##### Calculated by smax and Eit
    # 'eps_max_c': 'Maximum Strain (to Compressive Modulus)', ##### Calculated by smax and Eic
    # 'Machine',  ##### Not useful
    "control": "Control",  ##### Classification
    # 'grip', ##### Classification
    # 'ABG', ##### Classification
    "Temp.": "Temperature",
    # 'Temp. control', ##### Not useful
    # 'Environment', ##### Classification
    # 'Reference document',  ##### Not useful
    # 'Remarks', ##### Not useful
    # 'Invalid ', ##### Classification
    # 'Bending ', ##### Classification
    # 'Buckling ', ##### Classification
    # 'Overheating', ##### Classification
    # 'Tab failure ', ##### Classification
    # 'Delaminated ', ##### Classification
    # 'Incomplete measurement data ', ##### Classification
    # 'Strain from E ', ##### Classification
    # 'LUR', ##### Classification
    # 'TEC', ##### Too many absense
    # 'data delivered under name', ##### Not useful
    # 'Repair characteristics', ##### Classification
    # 'Strain measurement equipment (long)', ##### Not useful
    # 'Strain measurement equipment (short)', ##### Not useful
    # 'Grip pressure': 'Grip Pressure', ##### Not useful?
    "Geometry-tab thickness(for UD2; for MD2)": "Tab Thickness",
    # 'Plate-DSC analysis-Middle point Tg': 'DSC Analysis-Middle Point Tg',
    "Geometry-L": "Length",
    # 'Plate-Fibre content-Std.deviation W/W%', ##### Not useful
    # 'Laminate-from plate', ##### Same as Plate
    # 'Geometry-Repair-tape', ##### Classification
    # 'Plate-Density-Average': 'Plate Density',
    # 'Plate-Void content-Std. deviation', ##### Not useful
    # 'Geometry-Task Group',  ##### Classification
    # 'Laminate-Resin', ##### To be specified
    # 'Laminate-Fibre', ##### all the same
    # 'Laminate-Average thickness', ##### No value
    # 'Geometry-Repair-repair ratio', ##### How to use?
    "Plate-Fibre content-Average W/W%": "Fiber Weight Fraction",
    # 'Geometry-Manufacturer',  ##### Not useful
    # 'Geometry-Repair-Slope': 'Geometry Repair Slope', ##### Too many absence
    # 'Geometry-minimum\nwidth', ##### Too many absence
    # 'Plate-Resin content-Std. deviation', ##### Not useful
    # 'Geometry-Repair-repair material', ##### To be specified
    "Geometry-load\nlength": "Load Length",
    # 'Laminate-Description', ##### Not useful
    # 'Laminate-thickness (nominal)', ##### Better use thickness directly
    # 'Plate-Resin content-Average': 'Plate Resin Content',
    # 'Geometry-t(for UD2; for MD2)', ##### Better use thickness directly
    # 'Plate-DSC analysis-Final Tg': 'DSC Analysis-Final Point Tg',
    # 'Geometry-maximum\nwidth': 'Geometry Maximum Width', ##### Fill empty in Maximum/Minimum/Average Width
    "Laminate- 45°": "Layers of Fiber in 45-deg Direction",
    # 'Plate-Void content-Average': 'Plate Void Content', ##### Not useful
    # 'Laminate-lay-up': 'Lay-up', ##### How to use?
    # 'Plate-Density-Std.deviation', ##### Not useful
    # 'Laminate-Material', ##### Not useful
    # 'Geometry-Repair-depth': 'Geometry Repair Depth', ##### Too many absence
    # 'Plate-Fibre content-Std. Deviation V/V%', ##### Not useful
    # 'Plate-Laminate', ##### Same as Laminate
    "Laminate- 90°": "Layers of Fiber in 90-deg Direction",  ##### all zeros
    # 'Plate-Plate Tested by lab(s)', ##### No useful
    # 'Geometry-Repair-base material', ##### No useful
    # 'Plate-Fibre content-Average V/V%': 'Plate Fibre Content V/V%',
    # 'Geometry-width\n(cruciform arms)', ##### No useful Too many absence
    # 'Geometry-Repair-scarf/plug/patch', ##### No useful, Classification?
    # 'Laminate- -45°': 'Percentage of Fibre in -45-deg Direction',
    # 'Plate-Geometry/ies', ##### No useful
    # 'Laminate-Production method', ##### No useful
    "Laminate- 0°": "Layers of Fiber in 0-deg Direction",
    # 'Geometry-General Shape', ##### Classification?
    # 'Geometry-tab\nlength' ##### Not useful
    ################### Added features ###################
    "(Static) Maximum Tensile Stress": "Ultimate Tensile Stress",
    "(Static) Maximum Compressive Stress": "Ultimate Compressive Stress",
    "(Static) Maximum Tensile Strain": "Ultimate Tensile Strain",
    "(Static) Maximum Compressive Strain": "Ultimate Compressive Strain",
    "(Static) Eit": "Tensile Modulus",
    "(Static) Eic": "Compressive Modulus",
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
