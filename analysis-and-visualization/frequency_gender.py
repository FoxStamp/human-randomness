import matplotlib.pyplot as plt
import util

num_states = util.num_states

# load male and female data and generate control data
def load_and_generate_data():
    male_df = util.load_human_data("Please enter your gender", "Male")
    female_df = util.load_human_data("Please enter your gender", "Female")
    male_control_df = util.generate_random_data(male_df.shape)
    female_control_df = util.generate_random_data(male_df.shape)
    
    return male_df, female_df, male_control_df, female_control_df

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

male_df, female_df, male_control_df, female_control_df = load_and_generate_data()

# calculate maximum y-value + padding
ymax = util.calculate_ymax([util.count_occurrences(male_df),
                            util.count_occurrences(female_df),
                            util.count_occurrences(male_control_df),
                            util.count_occurrences(female_control_df)], 1.15)

fig, ax = plt.subplots(2, 2, figsize=(12, 7))

# plot male, female and controls
plot_bars(ax[0, 0], male_df, "blue", "Male Number Frequency")
plot_bars(ax[0, 1], female_df, "pink", "Female Number Frequency")
plot_bars(ax[1, 0], male_control_df, "gray", "Control Number Frequency with Male Dataset Size")
plot_bars(ax[1, 1], female_control_df, "gray", "Control Number Frequency with Female Dataset Size")

plt.tight_layout()
plt.show()