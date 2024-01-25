import pandas as pd
import matplotlib.pyplot as plt

csv_name = "Human Randomness_ Patterns and Predictability in Participant-Generated Number Sequences (Responses) - Dataset 1 (incomplete).csv"
df = pd.read_csv(csv_name)

nums_df = df.iloc[:, 1:51]

print(nums_df)

# nums_dict = {}
# for i in range(10):
#     nums_dict[i] = 0
