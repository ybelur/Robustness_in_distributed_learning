import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the data
file_path = "../results/median_data_poison.csv"
data = pd.read_csv(file_path)

data['Key Label'] = data['Number of Data Poisoned Clients'].apply(lambda x: f"{x} (No Data Poisoning)" if x == 0 else str(x))

# Set the plot style
sns.set(style="whitegrid")

# plt.figure(figsize=(12, 6))
# sns.lineplot(
#     data=data, 
#     x="Round", 
#     y="Global Accuracy", 
#     hue="Key Label", 
#     marker="o",
#     palette="bright"
# )

# plt.title("Global Accuracy vs Number of Rounds for Different Levels of Data Poisoning (Median Averaging)")
# plt.xlabel("Number of Rounds")
# plt.ylabel("Global Accuracy")
# plt.legend(title="Number of Poisoned Nodes", bbox_to_anchor=(1, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# Calculate the moving average
window = 5

data['Moving Average'] = data.groupby('Key Label')['Global Accuracy'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

# Plot the moving average
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=data, 
    x="Round", 
    y="Moving Average", 
    hue="Key Label", 
    marker="o",
    palette="bright"
)

plt.title("Moving Average of Global Accuracy vs Number of Rounds for Different Levels of Data Poisoning (Median Averaging)")
plt.xlabel("Number of Rounds")
plt.ylabel("Moving Average of Global Accuracy")
plt.legend(title="Number of Poisoned Nodes", bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.savefig("../figures/median_moving_avg_data_poison_10_clients_2.png", dpi=300)
plt.show()
