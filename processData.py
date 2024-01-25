import pandas as pd
import matplotlib.pyplot as plt

csv_name = "Human Randomness_ Patterns and Predictability in Participant-Generated Number Sequences (Responses) - Dataset 1 (incomplete).csv"
df = pd.read_csv(csv_name)

subset_df = df.iloc[:, 1:51]
count_data = subset_df.apply(lambda x: x.value_counts()).fillna(0)

count_data.plot(kind='bar', stacked=True, legend=False)

plt.xlabel('Number')
plt.ylabel('Count')
plt.title('Distribution across all data')

plt.show()