import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# List of uploaded CSV files and their corresponding poison levels
file_paths = [
    "../results/mean_data_poison/mean_data_poison_0_cx3.csv",
    "../results/mean_data_poison/mean_data_poison_0_cx3.csv",
    "../results/mean_data_poison/mean_data_poison_1_cx3.csv",
    "../results/mean_data_poison/mean_data_poison_2_cx3.csv",
    "../results/mean_data_poison/mean_data_poison_3_cx3.csv",
    "../results/mean_data_poison/mean_data_poison_4_cx3.csv",
    "../results/mean_data_poison/mean_data_poison_5_cx3.csv",
    "../results/mean_data_poison/mean_data_poison_6_cx3.csv",
    "../results/mean_data_poison/mean_data_poison_7_cx3.csv",
    "../results/mean_data_poison/mean_data_poison_8_cx3.csv",
    "../results/mean_data_poison/mean_data_poison_9_cx3.csv"
]

# Combine all data into a single DataFrame with moving averages
all_data = []

for path in file_paths:
    df = pd.read_csv(path)
    run = int(os.path.basename(path).split('_')[3]) + 1
    df['Trial Number'] = run
    # Calculate moving average (e.g., window size = 5)
    df['Global Accuracy'] = df['Global Accuracy'].rolling(window=5, min_periods=1).mean()
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

plt.title('Moving Average of Global Accuracy vs Round over several Trials (Mean Data Poisoning)')
plt.xlabel('Round')
plt.ylabel('Global Accuracy (Moving Average)')
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/mean_data_poison_prob_combined.png", dpi=300)
