import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#open human CSV
human_csv_name = "Human Randomness_ Patterns and Predictability in Participant-Generated Number Sequences (Responses) - Form Responses v2.csv"
human_df = pd.read_csv(human_csv_name).iloc[:, 1:51]

#open control CSV
control_csv_name = "random_50x50_seed_682.csv"
control_df = pd.read_csv(control_csv_name)

fig, ax = plt.subplots(5, 4)

# Loop over 10 possible numbers
for i in range(10):
    # Calculate the position in the subplot grid for the human dataset
    human_place = int(i/2), i%2

    control_place = human_place[0], human_place[1]+2

    # np arrays store count
    counts_human = np.zeros(10)
    counts_control = np.zeros(10)

    # Human df
    for _, row in human_df.iterrows():
        for j in range(50 - 1):
            if row[j] == i:
                counts_human[row[j + 1]] += 1

    # Control df
    for _, row in control_df.iterrows():
        for j in range(50 - 1):
            if row[j] == i:
                counts_control[row[j + 1]] += 1

    ax[human_place].bar(range(10), counts_human, color='green')
    ax[human_place].set_title(f"Human - Frequency Succeeding {i}",fontsize=8)

    ax[control_place].bar(range(10), counts_control, color='blue')
    ax[control_place].set_title(f"Control - Frequency Succeeding {i}",fontsize=8)

# Display plots
plt.tight_layout()
plt.show()