import os
import pandas as pd
from typing import List, Union


def read_lbs(paths: List[Union[os.PathLike, str]]) -> List[pd.DataFrame]:
    dfs = []
    for path in paths:
        df = pd.read_csv(path, index_col=0)
        dfs.append(df)
    return dfs


def merge_leaderboards(dfs: List[pd.DataFrame]):
    df = pd.concat(dfs, ignore_index=True)
    df.sort_values(
        by="Testing RMSE" if "Testing RMSE" in df.columns else "RMSE",
        ascending=True,
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df


def avg_rank(dfs: List[pd.DataFrame]):
    all_program_models = []
    each_program_models = []
    for df in dfs:
        each_program_models.append([(x, y) for x, y in zip(df["Program"], df["Model"])])
        all_program_models += each_program_models[-1]
    all_program_models = list(set(all_program_models))
    avg_df = pd.DataFrame(columns=["Program", "Model"])
    avg_df["Program"] = [x for x, y in all_program_models]
    avg_df["Model"] = [y for x, y in all_program_models]

    for df_idx, (df, program_models) in enumerate(zip(dfs, each_program_models)):
        for row_idx, (program, model) in enumerate(all_program_models):
            if (program, model) in program_models:
                idx = program_models.index((program, model))
                avg_df.loc[row_idx, f"Rank {df_idx}"] = list(df.index)[idx] + 1
    avg_df["Avg Rank"] = avg_df[[f"Rank {df_idx}" for df_idx in range(len(dfs))]].mean(
        axis=1
    )
    avg_df.sort_values(by="Avg Rank", ascending=True, inplace=True)
    avg_df.reset_index(drop=True, inplace=True)
    return avg_df


def merge_to_excel(
    path: Union[os.PathLike, str],
    dfs: List[pd.DataFrame],
    avg_df: pd.DataFrame,
    sheet_names: List[str] = None,
    **kwargs,
):
    avg_sheet_name = "Average"
    if sheet_names is None:
        sheet_names = [f"Mode {x}" for x in range(len(dfs))]
    elif len(sheet_names) == len(dfs) + 1:
        avg_sheet_name = sheet_names.pop(-1)
    with pd.ExcelWriter(path) as writer:
        for df, name in zip(dfs, sheet_names):
            df.to_excel(writer, sheet_name=name, **kwargs)
        avg_df.to_excel(writer, sheet_name=avg_sheet_name, **kwargs)
