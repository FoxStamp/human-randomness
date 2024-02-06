import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.stats import sem, t

# Open human CSV
human_csv_name = "Human Randomness Patterns and Predictability in Participant-Generated Number Sequences (Responses) - 50 count.csv"
human_df = pd.read_csv(human_csv_name)

female_df = human_df[human_df['Please enter your gender'] == 'Female'].iloc[:, 1:51]
male_df = human_df[human_df['Please enter your gender'] == 'Male'].iloc[:, 1:51]

#Random seed for reproducibility
random_seed = random.randint(1, 1000)
random.seed(random_seed)

# Make control Lists
control1_df = pd.DataFrame([[random.randint(0, 9) for _ in range(female_df.shape[1])] for _ in range(female_df.shape[0])])
control2_df = pd.DataFrame([[random.randint(0, 9) for _ in range(male_df.shape[1])] for _ in range(male_df.shape[0])])


fig, ax = plt.subplots(2, 2, figsize=(12, 7))

counts_female = np.zeros(10, dtype=int)
counts_male = np.zeros(10, dtype=int)
counts1_control = np.zeros(10, dtype=int)
counts2_control = np.zeros(10, dtype=int)

for index, row in male_df.items():
    for col_name, cell_value in row.items():
        counts_male[cell_value] += 1

for index, row in female_df.items():
    for col_name, cell_value in row.items():
        counts_female[cell_value] += 1

for index, row in control1_df.items():
    for col_name, cell_value in row.items():
        counts1_control[cell_value] += 1

for index, row in control2_df.items():
    for col_name, cell_value in row.items():
        counts2_control[cell_value] += 1

for row in ax:
    for col in row:
        col.set_ylabel("Percent of dataset")

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

plot_bar_with_confidence(ax[0, 0], counts_male, "Male Number Frequency", 'blue')
plot_bar_with_confidence(ax[0, 1], counts_female, "Female Number Frequency", 'pink')
plot_bar_with_confidence(ax[1, 0], counts2_control, "Control Number Frequency with Male Dataset Size", 'gray', control=True)
plot_bar_with_confidence(ax[1, 1], counts1_control, "Control Number Frequency with Female Dataset Size", 'gray', control=True)

plt.tight_layout()
plt.legend()
plt.show()