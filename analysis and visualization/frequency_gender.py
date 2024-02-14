import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.stats import sem, t
import yaml

# load config values
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

def generate_random_data(shape):
    return pd.DataFrame([[random.randint(0, 9) for _ in range(shape[1])] for _ in range(shape[0])])

csv_name = config['csv_name']
random_seed = config['random_seed']

# load and prepare dataframes
human_df = pd.read_csv(csv_name)

female_df = human_df[human_df['Please enter your gender'] == 'Female'].iloc[:, 1:51]
male_df = human_df[human_df['Please enter your gender'] == 'Male'].iloc[:, 1:51]

#Random seed for reproducibility
random.seed(random_seed)

# Make control Lists

control1_df = generate_random_data(female_df.shape)
control2_df = generate_random_data(male_df.shape)

def count_occurrences(dataframe):
    counts = np.zeros(10, dtype=int)
    for index, row in dataframe.items():
        for col_name, cell_value in row.items():
            counts[cell_value] += 1
    return counts

def plot_bar_with_confidence(ax, counts, title, color, control=False, random_seed=None):
    mean = np.array([x / len(counts) for x in counts])
    confidence = 0.95  # 95% confidence interval
    n = len(mean)
    error = sem(mean) * t.ppf((1 + confidence) / 2, n - 1)

    if control:
        ax.bar(range(10), mean, yerr=error, color=color, capsize=5, label=f"Seed: {random_seed}")
    else:
        ax.bar(range(10), mean, yerr=error, color=color, capsize=5)

    ax.set_title(title)
    ax.set_ylabel("Percent of dataset")

fig, ax = plt.subplots(2, 2, figsize=(12, 7))

counts_male = count_occurrences(male_df)
counts_female = count_occurrences(female_df)
counts1_control = count_occurrences(control1_df)
counts2_control = count_occurrences(control2_df)

for row in ax:
    for col in row:
        col.set_ylabel("Percent of dataset")

plot_bar_with_confidence(ax[0, 0], counts_male, "Male Number Frequency", 'blue')
plot_bar_with_confidence(ax[0, 1], counts_female, "Female Number Frequency", 'pink')
plot_bar_with_confidence(ax[1, 0], counts2_control, "Control Number Frequency with Male Dataset Size", 'gray', control=True)
plot_bar_with_confidence(ax[1, 1], counts1_control, "Control Number Frequency with Female Dataset Size", 'gray', control=True)

plt.tight_layout()
plt.legend()
plt.show()