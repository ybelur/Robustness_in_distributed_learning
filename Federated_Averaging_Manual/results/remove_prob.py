import os

# Path to your folder
folder_path = "median_prob_model_poison_7_clients/"

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if "dropout_mean" in filename:
        new_name = filename.replace("dropout_mean", "median")
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
