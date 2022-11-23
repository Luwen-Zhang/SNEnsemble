import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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
    fatigue_mat_lay = df_original['Material_Code'].copy()
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
            where_correlated = list(where_material[np.where(mse_matrix[i_idx, :] < 1e-5)[0]])
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
                df_final = pd.concat([df_final, df_original.loc[[cor[0]], :]], ignore_index=True, axis=0)
            else:
                pass  # Min Stress, Max Stress or frequency is not recorded
        bar.update(1)

    df_final.reset_index(drop=True, inplace=True)
    return df_final


def replace_column_name(df, name_mapping):
    columns = list(df.columns)
    for idx in range(len(columns)):
        try:
            columns[idx] = name_mapping[columns[idx]]
        except:
            pass
    df_tmp = df.copy()
    df_tmp.columns = columns
    return df_tmp


def code2seq(layer_original):
    pattern = re.compile(r'\d+(\(\d+\))')

    def bracket(layer):
        if '±' in layer:
            pm_indexes = []
            for x in range(len(layer)):
                if layer[x] == '±':
                    pm_indexes.append(x)
            for pm_idx in pm_indexes[::-1]:
                pm_value = layer[pm_idx + 1:].split('/')[0].split(']')[0]
                len_value = len(pm_value)
                pm_s = pm_value + '/' + '-' + pm_value
                layer = layer[:pm_idx] + pm_s + layer[pm_idx + len_value + 1:]

        if '[' not in layer and ']' not in layer:
            return layer

        queue = []
        for idx in range(len(layer)):
            if layer[idx] == '[':
                queue.append((1, idx))
            elif layer[idx] == ']':
                queue.append((2, idx))

        pairs = []  # pairs of brackets in the outer loop
        while len(queue) != 0 and queue[0][0] != 2:  # redundent right bracket left in the queue
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
                raise Exception('Not recognized.')
            pairs.append((first_idx, current_idx))

        if len(queue) == 1 and queue[0][0] == 2:
            # print('Warning', layer)
            if queue[0][1] == len(layer) - 1:
                layer = layer[:-1]
            else:
                layer = layer[:queue[0][1]] + layer[queue[0][1] + 1:]

        expanded = []
        for pair_idx, (left_bracket, right_bracket) in enumerate(pairs):
            q = bracket(layer[left_bracket + 1:right_bracket])
            if right_bracket != len(layer) - 1:
                postfix = layer[right_bracket + 1:].split('/')[0]
                # if the bracket is followed by 'S'
                inv = False
                if len(postfix) > 0 and (postfix[-1] == 's' or postfix[-1] == 'S'):
                    inv = True
                    postfix = postfix[:-1]
                try:
                    l = len(postfix)
                    n = round(float(postfix))
                    q = q.split('/') * n
                except:
                    l = 0
                    q = q.split('/')

                if inv:
                    q = q + q[::-1]

                last_place = right_bracket + 1 + l + int(inv)

                expanded.append('/'.join(q))
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

        return layer.strip(']').strip('[')

    layer = str(layer_original)

    # for FACT dataset
    layer = layer.replace('SB', '')
    layer = layer.replace('WR', '')
    layer = layer.replace('FW', '')
    layer = layer.replace('K', '')
    layer = layer.replace('()', '')
    layer = layer.replace('100CSM', '0')

    for match_str in re.findall(pattern, layer):
        layer = layer.replace(match_str, '')

    layer = layer.replace('s', 'S')
    layer = layer.replace('(', '[')
    layer = layer.replace(')', ']')
    layer = layer.replace('/FOAM/', ' | ')

    # for upwind dataset
    if (' - ' in layer and '/' in layer) or (' | ' in layer and '/' in layer):
        layer_tmp = layer.split(' - ') if ' - ' in layer else layer.split(' | ')
        layer_tmp = [x.split('/')[0] if x[-1] != ']' and x[-1] != 'S' else x.split('/')[0] + x[x.index(']'):] for x in
                     layer_tmp]
        layer = '/'.join(layer_tmp)

    layer = layer.replace(' - ', '/')  # for upwind dataset
    layer = layer.replace('|', '/')
    layer = layer.replace(',', '/')
    layer = layer.replace(';', '/')
    layer = layer.replace(' ', '')
    layer = layer.replace('\'', '')
    layer = layer.replace('°', '')
    layer = layer.replace('*', '')  # for upwind dataset
    layer = layer.replace('+/-', '±')
    layer = layer.replace('+-', '±')
    layer = layer.replace('RM', '0')  # RM and FOAM are treated as 0 for simplicity
    layer = layer.replace('C', '')  # C and G are for materials in SNL dataset
    layer = layer.replace('G', '')
    layer = layer.replace('M', '0')
    layer = layer.replace('N', '')
    layer = layer.replace('][', ']/[')
    # print(layer, end=' ' * (40 - len(layer)))
    q = bracket(layer)

    try:
        q = [int(x) for x in q.split('/')]
        return q
    except:
        return []  # Can not recognize the code


def plot_absence_ratio(ax, df_presence, **kargs):
    ax.set_axisbelow(True)
    x = df_presence["feature"].values
    y = df_presence["ratio"].values

    # ax.set_facecolor((0.97,0.97,0.97))
    # plt.grid(axis='x')
    plt.grid(axis="x", linewidth=0.2)
    # plt.barh(x,y, color= [clr_map[name] for name in x])
    sns.barplot(y, x, **kargs)
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
    plot_absence_ratio(ax, df_presence, orient='h', palette=clr, linewidth=1, edgecolor=[0, 0, 0])
    plt.tight_layout()

    plt.savefig(figure_name, dpi=600)
    # plt.close()
    # plt.show()
    plt.close()

def remove_s(x, s):
    if type(x) == str:
        if s in x:
            x = x.replace(s,'')
            x = float(x)
    return x

def cal_fraction(x, s):
    if type(x) == str:
        if s in x:
            x = x.split(s)
            x = (float(x[0])+float(x[1]))/2
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


def modify_col(df, column_name, func, **kargs):
    if column_name in list(df.columns):
        col = df[column_name]
        col = [func(x,**kargs) for x in col]
        df.loc[:, column_name] = col