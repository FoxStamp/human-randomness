import pandas as pd
import numpy as np
from tabulate import tabulate

# open human CSV
human_csv_name = "Form_Responses_v2.csv"
human_df = pd.read_csv(human_csv_name).iloc[:, 1:51]

# open control CSV
control_csv_name = "random_50x50_seed_682.csv"
control_df = pd.read_csv(control_csv_name)

# init counts
counts_human = np.zeros(10, dtype=int)
counts_control = np.zeros(10, dtype=int)

# count per num
for index, row in human_df.items():
    for col_name, cell_value in row.items():
        counts_human[cell_value] += 1

# count per num
for index, row in control_df.items():
    for col_name, cell_value in row.items():
        counts_control[cell_value] += 1

# calculate stats for human_df
human_stats = {
    "Mean": human_df.values.mean(),
    "Median": np.median(human_df.values),
    "Mode": human_df.stack().mode(),
    "Standard Deviation": human_df.values.flatten().std(),
    "Mean counts per num": counts_human.mean(),
    "Greatest difference between nums": counts_control.max()-counts_control.min()
}

# calculate stats for control_df
control_stats = {
    "Mean": control_df.values.mean(),
    "Median": np.median(control_df.values),
    "Mode": control_df.stack().mode(),
    "Standard Deviation": control_df.values.flatten().std(),
    "Mean counts per num": counts_control.mean(),
    "Greatest difference between nums:": counts_control.max()-counts_control.min()
}

# Create a tabulate table
table_data = [
    ["Human", human_stats["Mean"], human_stats["Median"], human_stats["Mode"], human_stats["Standard Deviation"], human_stats["Mean counts per num"], human_stats["Greatest difference between nums"]],
    ["Control", control_stats["Mean"], control_stats["Median"], control_stats["Mode"], control_stats["Standard Deviation"], human_stats["Mean counts per num"], human_stats["Greatest difference between nums"]]
]

headers = [key for key in human_stats.keys()]

table = tabulate(table_data, headers, tablefmt="grid")

# Print the table
print(table)