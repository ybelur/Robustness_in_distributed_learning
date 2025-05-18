# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os

# # List of uploaded CSV files and their corresponding poison levels
# file_paths = [
#     "../results/dropout_mean_data_poison/dropout_mean_data_poison_0_cx3.csv",
#     "../results/dropout_mean_data_poison/dropout_mean_data_poison_1_cx3.csv",
#     "../results/dropout_mean_data_poison/dropout_mean_data_poison_2_cx3.csv",
#     "../results/dropout_mean_data_poison/dropout_mean_data_poison_3_cx3.csv",
#     "../results/dropout_mean_data_poison/dropout_mean_data_poison_4_cx3.csv",
#     "../results/dropout_mean_data_poison/dropout_mean_data_poison_5_cx3.csv",
#     "../results/dropout_mean_data_poison/dropout_mean_data_poison_6_cx3.csv",
#     "../results/dropout_mean_data_poison/dropout_mean_data_poison_7_cx3.csv",
#     "../results/dropout_mean_data_poison/dropout_mean_data_poison_8_cx3.csv",
#     "../results/dropout_mean_data_poison/dropout_mean_data_poison_9_cx3.csv"
# ]

# # Combine all data into a single DataFrame with moving averages
# all_data = []

# for path in file_paths:
#     df = pd.read_csv(path)
#     run = int(os.path.basename(path).split('_')[4]) + 1
#     df['Trial Number'] = run
#     # Calculate moving average (e.g., window size = 5)
#     df['Global Accuracy'] = df['Global Accuracy'].rolling(window=5, min_periods=1).mean()
#     all_data.append(df)

# combined_df = pd.concat(all_data, ignore_index=True)

# # Create the seaborn line plot
# plt.figure(figsize=(12, 8))

# sns.lineplot(
#     data=combined_df,
#     x='Round', 
#     y='Global Accuracy', 
#     hue='Trial Number', 
#     palette='bright',
#     marker='o'
# )

# plt.title('Moving Average of Global Accuracy vs Rounds over several Trials (Dropout Mean Data Poisoning)')
# plt.xlabel('Round')
# plt.ylabel('Global Accuracy (Moving Average)')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("../figures/dropout_mean_data_poison_combined.png", dpi=300)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# List of uploaded CSV files and their corresponding poison levels
file_paths = [
    f"../results/dropout_median_0.5_prob_model_poison_5_clients/dropout_median_model_poison_cx3_run_{i:02d}.csv"
    for i in range(100)
]

# Combine all data into a single DataFrame with moving averages
all_data = []
max_accuracies = []

for path in file_paths:
    df = pd.read_csv(path)
    run = int(os.path.basename(path).split('_')[6].split('.')[0]) + 1
    df['Trial Number'] = run
    # Calculate moving average (e.g., window size = 5)
    df['Global Accuracy'] = df['Global Accuracy'].rolling(window=10, min_periods=1).mean()
    all_data.append(df)

    max_accuracies.append(df['Global Accuracy'].max())

combined_df = pd.concat(all_data, ignore_index=True)

# Create the seaborn line plot
plt.figure(figsize=(12, 8))

sns.lineplot(
    data=combined_df,
    x='Round', 
    y='Global Accuracy', 
    hue='Trial Number', 
    palette='bright',
    marker='o'
)

plt.title('Moving Average of Global Accuracy vs Rounds over several Trials (Dropout Median Model Poisoning)')
plt.xlabel('Round')
plt.ylabel('Global Accuracy (Moving Average)')
plt.grid(True)
plt.legend([], [], frameon=False) 
plt.savefig("../figures/dropout_median/5_clients/dropout_median_model_poison_100_prob.png", dpi=300)

# Compute and print statistics
max_accuracies = np.array(max_accuracies)
max_accuracies = max_accuracies[~np.isnan(max_accuracies)]

mean_accuracy = np.mean(max_accuracies)
median_accuracy = np.median(max_accuracies)
std_dev_accuracy = np.std(max_accuracies)
iqr_accuracy = np.percentile(max_accuracies, 75) - np.percentile(max_accuracies, 25)

print(f"Mean of highest global accuracies: {mean_accuracy:.4f}")
print(f"Median of highest global accuracies: {median_accuracy:.4f}")
print(f"Standard deviation of highest global accuracies: {std_dev_accuracy:.4f}")
print(f"Interquartile range (IQR) of highest global accuracies: {iqr_accuracy:.4f}")

# Plot the maximum accuracies as a scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(range(len(max_accuracies)), max_accuracies, color='blue', label='Max Accuracy')

# Add mean, median, and standard deviation lines
plt.axhline(mean_accuracy, color='green', linestyle='--', label=f'Mean: {mean_accuracy:.4f}')
plt.axhline(median_accuracy, color='orange', linestyle='--', label=f'Median: {median_accuracy:.4f}')
plt.axhline(mean_accuracy + std_dev_accuracy, color='red', linestyle='--', label=f'Mean + Std Dev: {mean_accuracy + std_dev_accuracy:.4f}')
plt.axhline(mean_accuracy - std_dev_accuracy, color='red', linestyle='--', label=f'Mean - Std Dev: {mean_accuracy - std_dev_accuracy:.4f}')

plt.title('Maximum Global Accuracy over Trials')
plt.xlabel('Trial Number')
plt.ylabel('Maximum Global Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()