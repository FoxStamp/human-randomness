import matplotlib.pyplot as plt
import util

num_states = util.num_states

# load human data and generate control data
def load_and_generate_data():
    human_df = util.load_human_data()
    control_df = util.generate_random_data(human_df.shape)
    
    return human_df, control_df

# plots bars with confidence interval
def plot_bars(ax, dataframe, color, title):
    proportion = util.count_occurrences(dataframe)

    ax.bar(range(num_states), proportion, yerr=util.proportion_ci(proportion, dataframe.size), color=color, capsize=5)
    ax.set_ylabel("Percent of dataset")
    ax.set_xlabel("Number")
    ax.set_title(title)
    ax.set_xticks(range(num_states))
    ax.set_xticklabels(range(num_states))
    ax.set_ylim(0, ymax)


human_df, control_df = load_and_generate_data()

# calculate maximum y-value + padding
ymax = util.calculate_ymax([util.count_occurrences(human_df),
                            util.count_occurrences(control_df)], 1.15)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# plot human and control
plot_bars(ax[0], human_df, "green", "Human")
plot_bars(ax[1], control_df, "blue", "Control")

fig.suptitle("Overall number frequency")

plt.tight_layout()
plt.show()