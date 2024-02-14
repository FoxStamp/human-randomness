import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import yaml

# load config values
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

num_states = config['num_states']
csv_name = config['csv_name']
random_seed = config['random_seed']

def generate_random_data(shape):
    return pd.DataFrame([[random.randint(0, 9) for _ in range(shape[1])] for _ in range(shape[0])])

def calculate_counts(data_df, num):
    counts = np.zeros(10)
    total_transitions = 0

    for _, row in data_df.iterrows():
        for j in range(50 - 1):
            if row[j] == num:
                counts[row[j + 1]] += 1
                total_transitions += 1

    # Convert counts to probabilities
    counts /= total_transitions

    return counts

def normalize_matrix(matrix):
    return matrix / np.sum(matrix, axis=1, keepdims=True)

def visualize_heatmap(ax, matrix, cmap, title):
    sns.heatmap(matrix, cmap=cmap, annot=True, fmt=".2f",
                xticklabels=range(num_states), yticklabels=range(num_states),
                vmin=0, vmax=max_prob, ax=ax)

    # Move ticks, tick labels, and x-axis title to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("Next State")
    ax.set_ylabel("Current State")
    ax.set_title(title)

# Open human CSV
human_df = pd.read_csv(csv_name).iloc[:, 1:51]

# Random seed for reproducibility
random.seed(random_seed)

# Make control Lists
control_df = generate_random_data(human_df.shape)

# Init counts for human and control datasets
prob_mat_human = np.array([calculate_counts(human_df, i) for i in range(num_states)])
prob_mat_control = np.array([calculate_counts(control_df, i) for i in range(num_states)])

# Normalize the rows of the matrix to obtain probabilities
prob_mat_human = normalize_matrix(prob_mat_human)
prob_mat_control = normalize_matrix(prob_mat_control)

# Get the maximum probability value from both matrices
max_prob = max(prob_mat_human.max(), prob_mat_control.max()) * 1.1

# Visualize the matrices using seaborn heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Human Dataset
visualize_heatmap(ax1, prob_mat_human, "Greens", "Human Dataset")

# Control Dataset
visualize_heatmap(ax2, prob_mat_control, "Blues", "Control Dataset")

plt.tight_layout()
plt.show()
