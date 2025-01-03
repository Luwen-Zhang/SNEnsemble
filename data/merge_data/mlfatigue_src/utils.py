import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import re
from pandas.api.types import is_numeric_dtype

clr = sns.color_palette("deep")


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

sns.reset_defaults()

matplotlib.rc("text", usetex=True)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["figure.autolayout"] = True


def averaging(df_original, measure_features):
    from sklearn.preprocessing import MinMaxScaler

    fatigue_mat_lay = df_original["Material_Code"].copy()
    df_final = pd.DataFrame(columns=df_original.columns)
    bar = tqdm(total=len(list(set(fatigue_mat_lay))))
    for material in list(set(fatigue_mat_lay)):
        where_material = np.where(fatigue_mat_lay == material)[0]
        # print(where_material)
        df_material = df_original.loc[where_material, measure_features].copy()
        scaler = MinMaxScaler()
        df_material.loc[:, :] = scaler.fit_transform(df_material)
        mse_matrix = np.zeros((len(where_material), len(where_material)))
        for i_idx, i in enumerate(where_material):
            for j_idx, j in enumerate(where_material):
                if j_idx < i_idx:
                    continue
                val_i = df_material.loc[i, :]
                val_j = df_material.loc[j, :]
                mse_val = np.mean((val_i - val_j) ** 2)
                mse_matrix[i_idx, j_idx] = mse_matrix[j_idx, i_idx] = mse_val

        all_correlation = []
        for i_idx, i in enumerate(where_material):
            where_correlated = list(
                where_material[np.where(mse_matrix[i_idx, :] < 1e-5)[0]]
            )
            if where_correlated not in all_correlation:
                all_correlation.append(where_correlated)
        # all_correlation = all_correlation
        for cor in all_correlation:
            if len(cor) > 1:
                df_avg = df_original.loc[[cor[0]], :].copy()
                for col in df_original.columns:
                    tmp = df_original.loc[cor, col]
                    if is_numeric_dtype(tmp):
                        df_avg[col] = np.mean(tmp)
                df_final = pd.concat([df_final, df_avg], ignore_index=True, axis=0)
            elif len(cor) == 1:
                df_final = pd.concat(
                    [df_final, df_original.loc[[cor[0]], :]], ignore_index=True, axis=0
                )
            else:
                pass  # Min Stress, Max Stress or frequency is not recorded
        bar.update(1)
    bar.close()
    df_final.reset_index(drop=True, inplace=True)
    return df_final


def replace_column_name(df, name_mapping):
    columns = list(df.columns)
    for idx in range(len(columns)):
        try:
            original_name = columns[idx]
            to_name = name_mapping[columns[idx]]
            if to_name in columns and original_name != to_name:
                warnings.warn(f"{name_mapping[columns[idx]]} already exist.")
            columns[idx] = to_name
        except:
            pass
    df_tmp = df.copy()
    df_tmp.columns = columns
    return df_tmp


def code2seq(layer_original):
    pattern = re.compile(r"\d+(\(\d+\))")

    def bracket(layer):
        if "±" in layer:
            pm_indexes = []
            for x in range(len(layer)):
                if layer[x] == "±":
                    pm_indexes.append(x)
            for pm_idx in pm_indexes[::-1]:
                pm_value = layer[pm_idx + 1 :].split("/")[0].split("]")[0]
                len_value = len(pm_value)
                pm_s = pm_value + "/" + "-" + pm_value
                layer = layer[:pm_idx] + pm_s + layer[pm_idx + len_value + 1 :]

        if "[" not in layer and "]" not in layer:
            return layer

        queue = []
        for idx in range(len(layer)):
            if layer[idx] == "[":
                queue.append((1, idx))
            elif layer[idx] == "]":
                queue.append((2, idx))

        pairs = []  # pairs of brackets in the outer loop
        while (
            len(queue) != 0 and queue[0][0] != 2
        ):  # redundent right bracket left in the queue
            t1, first_idx = queue.pop(0)
            current_idx = first_idx
            cnt_left_bracket = 1
            while cnt_left_bracket != 0:
                t, current_idx = queue.pop(0)
                if t == 1:
                    cnt_left_bracket += 1
                else:
                    cnt_left_bracket -= 1
            if first_idx == current_idx:
                raise Exception("Not recognized.")
            pairs.append((first_idx, current_idx))

        if len(queue) == 1 and queue[0][0] == 2:
            # print('Warning', layer)
            if queue[0][1] == len(layer) - 1:
                layer = layer[:-1]
            else:
                layer = layer[: queue[0][1]] + layer[queue[0][1] + 1 :]

        expanded = []
        for pair_idx, (left_bracket, right_bracket) in enumerate(pairs):
            q = bracket(layer[left_bracket + 1 : right_bracket])
            if right_bracket != len(layer) - 1:
                postfix = layer[right_bracket + 1 :].split("/")[0]
                # if the bracket is followed by 'S'
                inv = False
                if len(postfix) > 0 and (postfix[-1] == "s" or postfix[-1] == "S"):
                    inv = True
                    postfix = postfix[:-1]
                try:
                    l = len(postfix)
                    n = round(float(postfix))
                    q = q.split("/") * n
                except:
                    l = 0
                    q = q.split("/")

                if inv:
                    q = q + q[::-1]

                last_place = right_bracket + 1 + l + int(inv)

                expanded.append("/".join(q))
                if last_place == len(layer):
                    pairs[pair_idx] = (left_bracket, None)
                else:
                    pairs[pair_idx] = (left_bracket, last_place)
            else:
                expanded.append(q)
                pairs[pair_idx] = (left_bracket, None)

        for s, pair in zip(expanded[::-1], pairs[::-1]):
            left, right = pair
            if right is None:
                layer = layer[:left] + s
            else:
                layer = layer[:left] + s + layer[right:]

        return layer.strip("]").strip("[")

    layer = str(layer_original)

    # for FACT dataset
    layer = layer.replace("SB", "")
    layer = layer.replace("WR", "")
    layer = layer.replace("FW", "")
    layer = layer.replace("K", "")
    layer = layer.replace("()", "")
    layer = layer.replace("100CSM", "0")

    for match_str in re.findall(pattern, layer):
        layer = layer.replace(match_str, "")

    layer = layer.replace("s", "S")
    layer = layer.replace("(", "[")
    layer = layer.replace(")", "]")
    layer = layer.replace("/FOAM/", " | ")

    # for upwind dataset
    if (" - " in layer and "/" in layer) or (" | " in layer and "/" in layer):
        layer_tmp = layer.split(" - ") if " - " in layer else layer.split(" | ")
        layer_tmp = [
            (
                x.split("/")[0]
                if x[-1] != "]" and x[-1] != "S"
                else x.split("/")[0] + x[x.index("]") :]
            )
            for x in layer_tmp
        ]
        layer = "/".join(layer_tmp)

    layer = layer.replace(" - ", "/")  # for upwind dataset
    layer = layer.replace("|", "/")
    layer = layer.replace(",", "/")
    layer = layer.replace(";", "/")
    layer = layer.replace(" ", "")
    layer = layer.replace("'", "")
    layer = layer.replace("°", "")
    layer = layer.replace("*", "")  # for upwind dataset
    layer = layer.replace("+/-", "±")
    layer = layer.replace("+-", "±")
    layer = layer.replace("RM", "0")  # RM and FOAM are treated as 0 for simplicity
    layer = layer.replace("C", "")  # C and G are for materials in SNL dataset
    layer = layer.replace("G", "")
    layer = layer.replace("M", "0")
    layer = layer.replace("N", "")
    layer = layer.replace("][", "]/[")
    # print(layer, end=' ' * (40 - len(layer)))
    q = bracket(layer)

    try:
        q = [int(x) for x in q.split("/")]
        return q
    except:
        return []  # Can not recognize the code


def plot_absence_ratio(ax, df_presence, **kwargs):
    ax.set_axisbelow(True)
    x = df_presence["feature"].values
    y = df_presence["ratio"].values

    # ax.set_facecolor((0.97,0.97,0.97))
    # plt.grid(axis='x')
    plt.grid(axis="x", linewidth=0.2)
    # plt.barh(x,y, color= [clr_map[name] for name in x])
    sns.barplot(y, x, **kwargs)
    ax.set_xlim([0, 1])
    ax.set_xlabel("Data absence ratio")


def calculate_absence_ratio(df_tmp):
    df_presence = pd.DataFrame(columns=["feature", "ratio"])

    for column in df_tmp.columns:
        presence = len(np.where(df_tmp[column].notna())[0])
        # print(f'{column},\t\t {presence}/{len(df_all[column])}, {presence/len(df_all[column]):.3f}')

        df_presence = pd.concat(
            [
                df_presence,
                pd.DataFrame(
                    {"feature": column, "ratio": 1 - presence / len(df_tmp[column])},
                    index=[0],
                ),
            ],
            axis=0,
            ignore_index=True,
        )

    df_presence.sort_values(by="ratio", inplace=True, ascending=False)
    df_presence.reset_index(drop=True, inplace=True)

    # df_presence.drop([0, 1, 2, 5, 9, 10, 11, 13, 14, 15, 19, 23, ])

    return df_presence


def plot_absence(df_all, name_mapping, figure_name, fontsize=12):
    df_tmp = replace_column_name(df_all, name_mapping)
    col_to_include = [x for x in df_tmp.columns if x in name_mapping.values()]
    df_tmp = df_tmp[col_to_include]

    df_presence = calculate_absence_ratio(df_tmp)

    matplotlib.rc("text", usetex=False)
    plt.rcParams["font.size"] = fontsize

    clr = sns.color_palette("deep")

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    plot_absence_ratio(
        ax, df_presence, orient="h", palette=clr, linewidth=1, edgecolor=[0, 0, 0]
    )
    plt.tight_layout()

    plt.savefig(figure_name, dpi=600)
    # plt.close()
    # plt.show()
    plt.close()


def remove_s(x, s):
    if type(x) == str:
        if s in x:
            x = x.replace(s, "")
            x = float(x)
    return x


def cal_fraction(x, s):
    if type(x) == str:
        if s in x:
            x = x.split(s)
            x = (float(x[0]) + float(x[1])) / 2
    return x


def conditional_remove(x, s):
    if type(x) == str:
        if s in x:
            return np.nan
    return x


def conditional_replace(x, s1, s2):
    if type(x) == str:
        if s1 in x:
            x = x.replace(s1, s2)
    return x


def conditional_replace_strs(x, s1, s2):
    if type(x) == str:
        if s1 in x:
            return s2
    return x


def replace_strs(x, s1, s2):
    if type(x) == str:
        if x == s1:
            return s2
    return x


def str2num(x, s, n):
    if type(x) == str:
        if s == x:
            x = n
    return x


def remove_strs(x):
    if type(x) == str:
        return np.nan
    else:
        return x


def fill_na(x, n):
    if np.isnan(x):
        return n
    else:
        return x


def col_abs(x):
    return np.abs(x)


def col_neg(x):
    return -x


def modify_col(df, column_name, func, **kwargs):
    if column_name in list(df.columns):
        col = df[column_name]
        col = [func(x, **kwargs) for x in col]
        df.loc[:, column_name] = col


def merge_col(df, from_col, to_col):
    if from_col in list(df.columns) and to_col in list(df.columns):
        where = np.where(df[from_col].notna())[0]
        df.loc[where, to_col] = df.loc[where, from_col]
        del df[from_col]


def add_col(df, from_col, to_col):
    df[to_col] += df[from_col]


def main_process(
    df_all,
    name_database,
    name_mapping,
    modifications,
    unknown_categorical,
    na_to_0,
    merge,
    adds,
    force_remove_idx,
    init_r_col,
    init_max_stress_col,
    init_min_stress_col,
    init_max_strain_col,
    init_min_strain_col,
    init_seq_col,
    init_cycle_col,
    mat_code,
    static_to_extract,
    static_prefix,
    static_crit,
    static_merge_exist,
    fatigue_crit,
    not_runout_crit,
    illegal_crit,
    fill_absent_stress_r,
    eval_min_stress,
    extract_separate_e,
    get_static,
    special_treatments=None,
):
    # Plot absence first
    # plot_absence(
    #     df_all,
    #     name_mapping,
    #     f"../mlfatigue_output/{name_database}_absence_ratio_initial.png",
    #     fontsize=12,
    # )

    # Some indices are force removed
    df_all = df_all.loc[np.setdiff1d(df_all.index, force_remove_idx), :].copy()
    df_all.reset_index(drop=True, inplace=True)

    # Modify columns
    for col, op, kwargs in modifications:
        modify_col(df_all, col, op, **kwargs)
    # Fill Unknown categorical vars
    for col in unknown_categorical:
        df_all[col] = "UNK"

    # These columns use NaN to represent absence of fibre in the direction, so simply fillna by 0.
    for col in na_to_0:
        tmp = df_all[col].fillna(0)
        df_all.loc[:, col] = tmp

    # These columns have different names in different sheets
    for from_col, to_col in merge:
        merge_col(df_all, from_col, to_col)

    # These columns should be accumulated
    for from_col, to_col in adds:
        add_col(df_all, from_col, to_col)

    # Some static experiments do not have R-value and cycles to failure are 1.
    df_all.loc[np.where(static_crit(df_all))[0], init_r_col] = "static"

    static_indexes = np.where(df_all[init_r_col] == "static")[0]
    if fatigue_crit is None:
        non_static_indexes = np.setdiff1d(df_all.index, static_indexes)
    else:
        non_static_indexes = np.setdiff1d(
            np.where(fatigue_crit(df_all))[0], static_indexes
        )

    # if both min stress and max stress are given, and max stress is lower than min stress while both of them are negative
    if init_min_stress_col is not None:
        min_max_exist = np.intersect1d(
            np.where(1 - np.isnan(df_all[init_min_stress_col]))[0],
            np.where(1 - np.isnan(df_all[init_max_stress_col]))[0],
        )
        for idx in min_max_exist:
            max_s = df_all.loc[idx, init_max_stress_col]
            min_s = df_all.loc[idx, init_min_stress_col]
            if max_s < min_s < 0:
                df_all.loc[idx, init_max_stress_col] = min_s
                df_all.loc[idx, init_min_stress_col] = max_s
            elif max_s < min_s:
                warnings.warn(
                    f"Find max stress lower than min stress and not both of them are negative."
                )

    where_ge_1 = []
    where_le_1 = []
    where_le_1_ge_0 = []
    where_le_0 = []
    for idx, x in enumerate(df_all[init_r_col]):
        if not isinstance(x, str):
            if x > 1:
                where_ge_1.append(idx)
            elif x <= 1:  # ignore nan
                where_le_1.append(idx)
            if 0 <= x < 1:
                where_le_1_ge_0.append(idx)
            elif x < 0:
                where_le_0.append(idx)
    # If R>1, max stress/strain and min stress/strain must be negative
    df_all.loc[where_ge_1, init_max_stress_col] = -np.abs(
        df_all.loc[where_ge_1, init_max_stress_col]
    )
    if init_min_stress_col is not None:
        df_all.loc[where_ge_1, init_min_stress_col] = -np.abs(
            df_all.loc[where_ge_1, init_min_stress_col]
        )
    if init_max_strain_col is not None:
        df_all.loc[where_ge_1, init_max_strain_col] = -np.abs(
            df_all.loc[where_ge_1, init_max_strain_col]
        )
    if init_min_strain_col is not None:
        df_all.loc[where_ge_1, init_min_strain_col] = -np.abs(
            df_all.loc[where_ge_1, init_min_strain_col]
        )
    # Otherwise, max stress must be positive
    df_all.loc[where_le_1, init_max_stress_col] = np.abs(
        df_all.loc[where_le_1, init_max_stress_col]
    )
    if init_max_strain_col is not None:
        df_all.loc[where_le_1, init_max_strain_col] = np.abs(
            df_all.loc[where_le_1, init_max_strain_col]
        )
    # if 0 < R < 1, min stress/strain must be positive
    if init_min_stress_col is not None:
        df_all.loc[where_le_1_ge_0, init_min_stress_col] = np.abs(
            df_all.loc[where_le_1_ge_0, init_min_stress_col]
        )
    if init_min_strain_col is not None:
        df_all.loc[where_le_1_ge_0, init_min_strain_col] = np.abs(
            df_all.loc[where_le_1_ge_0, init_min_strain_col]
        )
    # if R < 0, min stress/strain must be negative
    if init_min_stress_col is not None:
        df_all.loc[where_le_0, init_min_stress_col] = -np.abs(
            df_all.loc[where_le_0, init_min_stress_col]
        )
    if init_min_strain_col is not None:
        df_all.loc[where_le_0, init_min_strain_col] = -np.abs(
            df_all.loc[where_le_0, init_min_strain_col]
        )

    if init_min_stress_col is not None and init_r_col is not None:
        # If R, min stress, and max stress are all presented, check whether their relationships are correct.
        all_presented = np.intersect1d(
            np.intersect1d(
                np.where(1 - np.isnan(df_all[init_max_stress_col]))[0],
                np.where(1 - np.isnan(df_all[init_min_stress_col]))[0],
            ),
            np.where(1 - pd.isna(df_all[init_r_col]))[0],
        )
        for idx in all_presented:
            if idx in non_static_indexes and type(df_all.loc[idx, init_r_col]) != str:
                target = np.abs(
                    df_all.loc[idx, init_min_stress_col]
                    / df_all.loc[idx, init_max_stress_col]
                    - df_all.loc[idx, init_r_col]
                ) / np.abs(df_all.loc[idx, init_r_col])
                if target > 1:  # only fix absurd values.
                    r = round(
                        df_all.loc[idx, init_min_stress_col]
                        / df_all.loc[idx, init_max_stress_col],
                        1,
                    )
                    df_all.loc[idx, init_r_col] = r
                elif 0.1 < target <= 1:
                    df_all.loc[idx, init_min_stress_col] = np.nan

    # Fill max/min stress or R-value using each other.
    if fill_absent_stress_r:
        miss_max_stress_indexes = np.where(np.isnan(df_all[init_max_stress_col]))[0]
        miss_min_stress_indexes = np.where(np.isnan(df_all[init_min_stress_col]))[0]
        miss_R_value_indexes = np.where(pd.isna(df_all[init_r_col]))[0]

        for idx in miss_max_stress_indexes:
            if idx in non_static_indexes and type(df_all.loc[idx, init_r_col]) != str:
                df_all.loc[idx, init_max_stress_col] = (
                    df_all.loc[idx, init_min_stress_col] / df_all.loc[idx, init_r_col]
                )

        for idx in miss_min_stress_indexes:
            if idx in non_static_indexes and type(df_all.loc[idx, init_r_col]) != str:
                df_all.loc[idx, init_min_stress_col] = (
                    df_all.loc[idx, init_max_stress_col] * df_all.loc[idx, init_r_col]
                )

        for idx in miss_R_value_indexes:
            if idx in non_static_indexes:
                df_all.loc[idx, init_r_col] = (
                    df_all.loc[idx, init_min_stress_col]
                    / df_all.loc[idx, init_max_stress_col]
                )

    if eval_min_stress is not None:
        df_all.loc[non_static_indexes, eval_min_stress] = (
            df_all.loc[non_static_indexes, init_max_stress_col]
            * df_all.loc[non_static_indexes, init_r_col]
        )

    # Extract Compressive/Elastic Modulus
    if extract_separate_e is not None:
        for item in extract_separate_e:
            init_col = item["init_col"]
            comp_name = item["comp_name"]
            tensile_name = item["tensile_name"]
            comp_crit = item["comp_crit"]
            df_all[tensile_name] = np.nan
            df_all[comp_name] = np.nan
            for idx in range(len(df_all)):
                if comp_crit(df_all.loc[idx, :]):
                    df_all.loc[idx, comp_name] = df_all.loc[idx, init_col]
                else:
                    df_all.loc[idx, tensile_name] = df_all.loc[idx, init_col]

    # Add material code
    material_code = mat_code(df_all)
    df_all["Material_Code"] = material_code

    # Translate laminate code
    if init_seq_col is not None:
        if init_seq_col in df_all.columns:
            code2seq_dict = {}
            layups = df_all[init_seq_col].values
            for layer in list(set(layups)):
                code2seq_dict[layer] = code2seq(layer)
            seq = []
            for layer in layups:
                seq.append("/".join([str(x) for x in code2seq(layer)]))
            df_all["Sequence"] = seq

    # Find static and non_static data separately
    df_tmp = df_all.copy()

    df_fatigue = df_tmp.loc[non_static_indexes].copy()
    df_fatigue.reset_index(drop=True, inplace=True)

    if get_static:
        df_static = df_tmp.loc[static_indexes].copy()
        df_static.reset_index(drop=True, inplace=True)
        replace_column_name(df_static, name_mapping).to_excel(
            f"../mlfatigue_output/{name_database}_static.xlsx",
            engine="openpyxl",
            index=False,
        )

        ### Extract material properties from static experiments
        static_mat_lay = df_static["Material_Code"].copy()
        static_properties = {}

        static_features = static_to_extract

        for material in list(set(static_mat_lay)):
            where_material = np.where(static_mat_lay == material)[0]
            # print(material, len(where_material))
            material_data = df_static.loc[where_material, static_features].copy()
            material_data.reset_index(drop=True, inplace=True)
            material_df = {}
            for feature in static_features:
                for idx in range(len(material_data[feature])):
                    if type(material_data.loc[idx, feature]) == str:
                        material_data.loc[idx, feature] = np.nan

                presence_indexes = np.where(material_data[feature])[0]
                mean_value = np.mean(material_data.loc[presence_indexes, feature])
                material_df[feature] = mean_value

            material_df = pd.DataFrame(material_df, index=[0])
            static_properties[material] = material_df

        fatigue_static_features = [
            static_prefix + x if static_prefix not in x else x for x in static_features
        ]

        fatigue_mat_lay = df_fatigue["Material_Code"].copy()

        for fat_feature, sta_feature in zip(fatigue_static_features, static_features):
            if fat_feature not in list(df_fatigue.columns):
                df_fatigue[fat_feature] = np.nan
            if static_merge_exist is not None and sta_feature in static_merge_exist:
                df_fatigue[fat_feature] = df_fatigue[sta_feature]

        for material in list(set(static_mat_lay)):
            where_material = np.where(fatigue_mat_lay == material)[0]
            if len(where_material) > 0:
                static_property = static_properties[material]
                for fat_feature, sta_feature in zip(
                    fatigue_static_features, static_features
                ):
                    feature_absence = np.where(pd.isna(df_fatigue[fat_feature]))[0]
                    to_assign = np.intersect1d(where_material, feature_absence)
                    df_fatigue.loc[to_assign, fat_feature] = static_property[
                        sta_feature
                    ].values[0]

        # plot_absence(
        #     df_static,
        #     name_mapping,
        #     f"../mlfatigue_output/{name_database}_static_absence_ratio.png",
        #     fontsize=12,
        # )

    # Replace illegal to nan
    for col, crit in illegal_crit.items():
        df_fatigue.loc[np.where(crit(df_fatigue))[0], col] = np.nan

    # Remove runouts
    if not_runout_crit is not None:
        df_fatigue = df_fatigue.loc[np.where(not_runout_crit(df_fatigue))[0], :].copy()
        df_fatigue.reset_index(drop=True, inplace=True)

    if special_treatments is not None:
        for treatment in special_treatments:
            df_fatigue = treatment(df_fatigue)

    df_fatigue["log10(Cycles to Failure)"] = np.log10(
        df_fatigue[init_cycle_col].values.astype(float)
    )
    replace_column_name(df_fatigue, name_mapping).to_excel(
        f"../mlfatigue_output/{name_database}_fatigue.xlsx",
        engine="openpyxl",
        index=False,
    )

    # plot_absence(
    #     df_fatigue,
    #     name_mapping,
    #     f"../mlfatigue_output/{name_database}_absence_ratio.png",
    #     fontsize=12,
    # )
