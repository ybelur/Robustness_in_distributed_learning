import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('no_attack_results.csv')

# Plot the data
plt.figure(figsize=(12, 6))

# Iterate over each column (except the first one if it's "Rounds")
for column in data.columns[1:]:
    plt.plot(data['Rounds'], data[column], label=column)

# Customize the plot
plt.title('Raw Accuracy over 70 Rounds for Different Epochs per Round')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.legend(title='Epochs per Round')
plt.grid(True)

# Save the plot
plt.savefig('accuracy_plot.png')

# Display the plot
plt.show()
