from src.utils.ranking import *

dfs = read_lbs(
    [
        "output/composite_database_03012023/2023-04-04-11-40-06-0_composite_MC_10repeats_bayes/leaderboard.csv",
        "output/composite_database_03012023/2023-04-16-14-37-06-0_composite_SNTrans_MC/leaderboard.csv",
    ]
)
df_mc = merge_leaderboards(dfs)
dfs = read_lbs(
    [
        "output/composite_database_03012023/2023-04-12-22-37-51-0_composite_C_10repeats_bayes/leaderboard.csv",
        "output/composite_database_03012023/2023-04-16-20-38-08-0_composite_SNTrans_C/leaderboard.csv",
    ]
)
df_c = merge_leaderboards(dfs)
dfs = read_lbs(
    [
        "output/composite_database_03012023/2023-04-05-20-56-17-0_composite_M_10repeats_bayes/leaderboard.csv",
        "output/composite_database_03012023/2023-04-16-17-42-21-0_composite_SNTrans_M/leaderboard.csv",
    ]
)
df_m = merge_leaderboards(dfs)
dfs = read_lbs(
    [
        "output/composite_database_03012023/2023-04-07-18-22-05-0_composite_R_10repeats_bayes/leaderboard.csv",
        "output/composite_database_03012023/2023-04-17-00-37-02-0_composite_SNTrans_R/leaderboard.csv",
    ]
)
df_r = merge_leaderboards(dfs)
df = avg_rank([df_mc, df_c, df_m, df_r])
merge_to_excel(
    path="output/summary.xlsx",
    dfs=[df_mc, df_c, df_m, df_r],
    avg_df=df,
    sheet_names=[
        "Material-cycle extrap.",
        "Cycle extrap.",
        "Material extrap.",
        "Random",
        "Average Ranking",
    ],
    index=False,
    engine="openpyxl",
)
