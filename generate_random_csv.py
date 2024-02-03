import csv
import random

#Random seed for reproducibility
random_seed = random.randint(1, 1000)
random.seed(random_seed)

# Size of CSV grid
rows = 50
cols = 50

# Generate vals (0-9)
data = [[random.randint(0, 9) for _ in range(cols)] for _ in range(rows)]

# CSV file name
csv_filename = f"random_{rows}x{cols}_seed_{random_seed}.csv"

# Write data
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"Random CSV file '{csv_filename}' created successfully with seed {random_seed}.")
