import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = "results/no_attack_results.csv"
data = pd.read_csv(file_path)

# Plot 1: Raw Accuracy
plt.figure(figsize=(12, 5))
for epochs in sorted(data['Number of Epochs'].unique()):
    subset = data[data['Number of Epochs'] == epochs]
    plt.plot(subset['Round'], subset['Global Accuracy'], label=f'{epochs} Epochs per Round')
plt.title('Raw Accuracy over 70 Rounds for Different Epochs per Round')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('figures/no_attack_raw_accuracy.png', dpi=300)
# plt.show()

plt.figure(figsize=(12, 5))
window_size = 10  # Window size for moving average
for epochs in sorted(data['Number of Epochs'].unique()):
    subset = data[data['Number of Epochs'] == epochs]
    moving_avg = subset['Global Accuracy'].rolling(window=window_size).mean()
    plt.plot(subset['Round'], moving_avg, label=f'{epochs} Epochs (Moving Avg)')

plt.title('Moving Average of Accuracy over 70 Rounds for Different Epochs per Round')
plt.xlabel('Rounds')
plt.ylabel('Accuracy (Moving Avg)')
plt.legend()
plt.grid(True)
plt.savefig('figures/no_attack_moving_avg_accuracy.png', dpi=300)
# plt.show()
