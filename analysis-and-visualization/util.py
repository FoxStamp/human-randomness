import pandas as pd
import numpy as np
import random
from scipy.stats import norm
import yaml

# load config values
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

num_states = config['num_states']
csv_name = config['csv_name']
random_seed = config['random_seed']
chunk_size = config['chunk_size']

random.seed(random_seed)

def calculate_ymax(mats, coefficient):
    if not mats:
        raise ValueError("Input list is empty")
    
    ymax = 0

    for mat in mats:
        ymax = max(ymax, np.amax(mat))

    return ymax * coefficient

def calculate_proportion_ci(sample_proportion: float, total_observation: int, confidence: float) -> float:

    z_score = norm.ppf((1 + confidence) / 2)
    yerr = z_score * np.sqrt((sample_proportion * (1 - sample_proportion)) / total_observation)
    
    return yerr

def proportion_ci(proportions: np.ndarray, size: int, confidence: float = 0.95) -> np.ndarray:
    if size <= 0:
        raise ValueError("Size should be greater than 0")
    if not (0 < confidence < 1):
        raise ValueError("Confidence level should be between 0 and 1 exclusive")

    return np.array([calculate_proportion_ci(x,size,confidence) for x in proportions])

def generate_random_data(shape: tuple) -> pd.DataFrame:
    if not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError("Input shape should be a tuple of positive integers")

    return pd.DataFrame([[random.randint(0, 9) for _ in range(shape[1])] for _ in range(shape[0])])

def count_occurrences(dataframe: pd.DataFrame) -> np.ndarray:
    counts = np.zeros(num_states)

    for _, row in dataframe.iterrows():
        for cell_value in row:
            counts[cell_value] += 1

    counts /= dataframe.size

    return counts

def count_transitions(dataframe: pd.DataFrame, num: int) -> np.ndarray:
    if not (0 <= num < num_states):
        raise ValueError("Number should be in the range of valid states")

    counts = np.zeros(num_states)
    total_transitions = 0

    for _, row in dataframe.iterrows():
        for j in range(len(row) - 1):
            if row[j] == num:
                counts[row[j + 1]] += 1
                total_transitions += 1

    # Convert counts to probabilities
    counts /= total_transitions

    return counts

def prob_mat(dataframe: pd.DataFrame) -> np.ndarray:
    return np.array([count_transitions(dataframe, i) for i in range(num_states)])

def load_human_data(sortRow: str = None, value: str = None) -> pd.DataFrame:
    human_df = pd.read_csv(csv_name)

    if sortRow:
        if value:
            return human_df[human_df[sortRow] == value].iloc[:, 1:51]
        
        raise ValueError("No value inputted to sort by")

    
    return human_df.iloc[:, 1:51]