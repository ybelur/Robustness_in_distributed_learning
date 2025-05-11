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

# List of uploaded CSV files and their corresponding poison levels
file_paths = [
    f"../results/dropout_mean_data_poison_batch/dropout_mean_data_poison_cx3_run_{i:02d}.csv"
    for i in range(100)
]

# Combine all data into a single DataFrame with moving averages
all_data = []

for path in file_paths:
    df = pd.read_csv(path)
    run = int(os.path.basename(path).split('_')[6].split('.')[0]) + 1
    df['Trial Number'] = run
    # Calculate moving average (e.g., window size = 5)
    df['Global Accuracy'] = df['Global Accuracy'].rolling(window=10, min_periods=1).mean()
    all_data.append(df)

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

plt.title('Moving Average of Global Accuracy vs Rounds over several Trials (Dropout Mean Data Poisoning)')
plt.xlabel('Round')
plt.ylabel('Global Accuracy (Moving Average)')
plt.grid(True)
plt.legend([], [], frameon=False) 
plt.savefig("../figures/dropout_mean_data_poison_combined_100.png", dpi=300)
