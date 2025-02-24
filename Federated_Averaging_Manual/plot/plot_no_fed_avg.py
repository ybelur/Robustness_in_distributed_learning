import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = "../results/no_fed_avg.csv"
data = pd.read_csv(file_path)

# Plot the data
sns.set(style="whitegrid")

# plt.figure(figsize=(10, 6))
# sns.lineplot(
#     data=data,
#     x="Epoch",
#     y="Accuracy",
#     marker="o",
#     palette="bright"
# )

# plt.title("Global Accuracy vs Number of Rounds")
# plt.xlabel("Number of Rounds")
# plt.ylabel("Global Accuracy")
# plt.legend(title="Number of Epochs per Round", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.savefig("../figures/no_fed_avg.png", bbox_inches='tight', dpi=300)

# Calculate the moving average

data["Moving Average"] = data["Accuracy"].rolling(window=5, min_periods=1).mean()
plt.figure()

sns.lineplot(
    data=data,
    x="Epoch",
    y="Moving Average",
    label="Moving Average",
    marker="o",
    palette="bright"
)

plt.title("Global Accuracy vs Number of Rounds")
plt.xlabel("Number of Rounds")
plt.ylabel("Global Accuracy")
plt.legend(title="Number of Epochs per Round", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()
plt.savefig("../figures/no_fed_avg_moving_avg.png", bbox_inches='tight', dpi=300)
