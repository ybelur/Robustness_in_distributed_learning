import pandas as pd
import os
import glob

# Set your directory path where the CSV files are stored
folder_path = 'median_model_poison/'

# Pattern to match all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Read and combine all CSV files
combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Save to a new CSV file
combined_df.to_csv('combined_output.csv', index=False)
