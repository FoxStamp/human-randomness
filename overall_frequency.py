import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem, t

# Open human CSV
human_csv_name = "Form_Responses_v2.csv"
human_df = pd.read_csv(human_csv_name).iloc[:, 1:51]

# Open control CSV
control_csv_name = "random_50x50_seed_682.csv"
control_df = pd.read_csv(control_csv_name)

# Get random seed from file name
random_seed = control_csv_name[ control_csv_name.find('seed_') + 5 : control_csv_name.find('.csv')]

fig, ax = plt.subplots(1, 2, figsize=(10, 5)) # change size as necessary

# Calculate confidence interval and plot
def plot_bar_with_confidence(ax, counts, title, color, control=False):
    mean = np.array([x/human_df.size for x in counts])
    confidence = 0.95  # 95% confidence interval
    n = len(mean)
    error = sem(mean) * t.ppf((1 + confidence) / 2, n - 1)

    if control:
        ax.bar(range(10), mean, yerr=error, color=color, capsize=5, label=f"Seed: {random_seed}")
    else:
        ax.bar(range(10), mean, yerr=error, color=color, capsize=5)
    
    ax.set_title(title)
    ax.set_ylabel("Percent of dataset")

# numpy arrays to store counts
counts_human = np.zeros(10, dtype=int)
counts_control = np.zeros(10, dtype=int)

# count human data
for index, row in human_df.items():
    for col_name, cell_value in row.items():
        counts_human[cell_value] += 1

# count control data
for index, row in control_df.items():
    for col_name, cell_value in row.items():
        counts_control[cell_value] += 1

plot_bar_with_confidence(ax[0], counts_human, "Overall Number Frequency", 'green')
plot_bar_with_confidence(ax[1], counts_control, "Control Number Frequency", 'gray', control=True)

tallest_bar = max(max(counts_human), max(counts_control)) / human_df.size
ax[0].set_ylim(0, tallest_bar * 1.15)
ax[1].set_ylim(0, tallest_bar * 1.15)

# output_file = "figures\Figure_1c.png"
# dpi = 300  # Adjust the DPI as needed
# plt.savefig(output_file, dpi=dpi)

plt.tight_layout()
plt.show()