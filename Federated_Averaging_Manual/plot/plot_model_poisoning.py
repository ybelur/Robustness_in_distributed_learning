import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("../results/model_poison_results_5_clients.csv")

# Configure Seaborn for better aesthetics
sns.set(style="whitegrid")

# Update the 'Scale Factor' column for display
data['Scale Factor Label'] = data['Scale Factor'].apply(lambda x: f"{x} (No Model Poisoning)" if x == 1 else str(x))

# # Plot the accuracy against the number of rounds for different scale factors
# plt.figure(figsize=(12, 8))
# sns.lineplot(
#     data=data, 
#     x="Round", 
#     y="Global Accuracy", 
#     hue="Scale Factor Label", 
#     marker="o",
#     palette="bright"
# )

# # Add plot labels and title
# plt.title("Global Accuracy vs Number of Rounds for Different Scale Factors")
# plt.xlabel("Number of Rounds")
# plt.ylabel("Global Accuracy")
# plt.legend(title="Scale Factor")
# plt.savefig("../figures/model_poisoning_plot.png", dpi=300)

# Calculate the moving average
window = 5
data['Moving Average'] = data.groupby('Scale Factor')['Global Accuracy'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

# Plot the moving average of accuracy against the number of rounds for different scale factors
plt.figure(figsize=(12, 8))
sns.lineplot(
    data=data, 
    x="Round", 
    y="Moving Average", 
    hue="Scale Factor Label", 
    marker="o",
    palette="bright"
)

# Add plot labels and title
plt.title("Global Accuracy (Moving Average) vs Number of Rounds for Different Scale Factors")
plt.xlabel("Number of Rounds")
plt.ylabel("Global Accuracy (Moving Average)")
plt.legend(title="Scale Factor")
plt.savefig("../figures/model_poisoning_moving_avg_plot_5_clients.png", dpi=300)
