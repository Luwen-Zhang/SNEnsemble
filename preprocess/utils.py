import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import re

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
