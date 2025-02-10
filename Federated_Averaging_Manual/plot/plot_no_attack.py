import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = "../results/no_attack_results.csv"
data = pd.read_csv(file_path)


# Plot the data
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=data,
    x="Round",
    y="Global Accuracy",
    hue="Number of Epochs",
    marker="o",
    palette="bright"
)

plt.title("Global Accuracy vs Number of Rounds")
plt.xlabel("Number of Rounds")
plt.ylabel("Global Accuracy")
plt.legend(title="Number of Epochs per Round")
plt.savefig("../figures/no_attack_plot.png")

# Calculate the moving average
window = 10
data['Moving Average'] = data.groupby('Number of Epochs')['Global Accuracy'].transform(lambda x: x.rolling(window = window, min_periods=10).mean())

# Plot the moving average
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=data,
    x="Round",
    y="Moving Average",
    hue="Number of Epochs",
    marker="o",
    palette="bright"
)

plt.title("Moving Average of Global Accuracy vs Number of Rounds")
plt.xlabel("Number of Rounds")
plt.ylabel("Moving Average of Global Accuracy")
plt.legend(title="Number of Epochs per Round")
plt.savefig("../figures/no_attack_moving_average_plot.png")
