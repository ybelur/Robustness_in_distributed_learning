import os

# Path to your folder
folder_path = "dropout_median_0.5_prob_model_poison_5_clients/"

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if "median" in filename:
        new_name = filename.replace("median", "dropout_median")
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
