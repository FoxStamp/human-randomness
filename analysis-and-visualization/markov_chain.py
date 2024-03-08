import util
import numpy as np

# get num states
num_states = util.num_states

# load human data and generate control data
def load_and_generate_data():
    human_df = util.load_human_data()
    control_df = util.generate_random_data(human_df.shape)
    return human_df, control_df

# markov chain using probability matrix
def markov_chain(init_state, transition_matrix, sequence_length):
    sequence = [init_state]
    
    for i in range(sequence_length-1):
        next_state = np.random.choice(range(num_states), p=transition_matrix[sequence[i]])
        sequence.append(next_state)

    return sequence

human_df, control_df = load_and_generate_data()

# calculate transition probability matrices
prob_mat_human = util.prob_mat(human_df)
prob_mat_control = util.prob_mat(control_df)

print("Human Markov Chains:")

for i in range(10):
    # random initial state
    initial_state = np.random.randint(0, num_states - 1)
    sequence_length = 10 # change as necessary
    print(markov_chain(initial_state, prob_mat_human, sequence_length))