import os

# Path to your folder
folder_path = "dropout_mean_0.5_prob_model_poison_batch/"

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if "0.5_prob_" in filename:
        new_name = filename.replace("0.5_prob_", "")
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
