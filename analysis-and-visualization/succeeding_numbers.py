import matplotlib.pyplot as plt
import seaborn as sns
import util

num_states = util.num_states

def load_and_generate_data():
    human_df = util.load_human_data()
    control_df = util.generate_random_data(human_df.shape)
    
    return human_df, control_df


def visualize_heatmap(ax, matrix, color, title):
    sns.heatmap(matrix, cmap=color, annot=True, fmt=".2f",
                xticklabels=range(num_states), yticklabels=range(num_states),
                vmin=0, vmax=max_prob, ax=ax)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("Next State")
    ax.set_ylabel("Current State")
    ax.set_title(title)

human_df, control_df = load_and_generate_data()

prob_mat_human = util.prob_mat(human_df)
prob_mat_control = util.prob_mat(control_df)

max_prob = util.calculate_ymax([prob_mat_control,prob_mat_human], coefficient=1.1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

visualize_heatmap(ax1, prob_mat_human, "Greens", "Human Dataset")
visualize_heatmap(ax2, prob_mat_control, "Blues", "Control Dataset")

plt.tight_layout()
plt.show()