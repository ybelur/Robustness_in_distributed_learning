import os

# Path to your folder
folder_path = "weighted_mean_model_poison_batch/"

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if "prob_" in filename:
        new_name = filename.replace("prob_", "")
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
