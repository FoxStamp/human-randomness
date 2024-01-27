import pandas as pd
import matplotlib.pyplot as plt

csv_name = "Human Randomness_ Patterns and Predictability in Participant-Generated Number Sequences (Responses) - 50 count.csv"
human_df = pd.read_csv(csv_name)

subset_human_df = human_df.iloc[:, 1:51]


plt.xlabel('Number')
plt.ylabel('Count')
plt.title('Distribution across all data')

plt.show()