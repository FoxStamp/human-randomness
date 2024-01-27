import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#open human CSV
csv_name = "Human Randomness_ Patterns and Predictability in Participant-Generated Number Sequences (Responses) - 50 count.csv"
human_df = pd.read_csv(csv_name)

#select only the numbers
subset_human_df = human_df.iloc[:, 1:51]

fig, ax = plt.subplots(5, 2)

#for numbers 0-9 count each
for i in range(10):
    place = int(i/2), i%2

    counts = np.zeros(10)

    for _, row in subset_human_df.iterrows():
        for j in range(50 - 1):
            if row[j] == i:
                counts[row[j + 1]] += 1

    ax[place].bar(range(10), counts, color='green')
    ax[place].set_title(f"Nums succeeding {i}")


plt.tight_layout()
plt.show()