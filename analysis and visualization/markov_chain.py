import numpy as np
import pandas as pd
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

def simulate_markov_chain(current_state, transition_matrix, sequence_length, sequence=None):
    if sequence is None:
        sequence = [current_state]

    if len(sequence) == sequence_length:
        return sequence
    else:
        next_state = np.random.choice(range(num_states), p=transition_matrix[current_state])
        print(f"Current State: {current_state}, Next State: {next_state}, Probability: \n{transition_matrix[current_state]}")
        sequence.append(next_state)
        return simulate_markov_chain(next_state, transition_matrix, sequence_length, sequence)


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

# Simulate Markov Chains
print("Human Markov Chains:")

initial_state = random.randint(0, num_states - 1)
markov_chain_length = 10

print(simulate_markov_chain(initial_state, prob_mat_human, markov_chain_length))
