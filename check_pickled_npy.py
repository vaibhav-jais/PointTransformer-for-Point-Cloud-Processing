import os
import numpy as np
import json
import sys
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.insert(0, ROOT_DIR)


machine_config_json_path = os.path.join("configs", "machine_configs", "machine_config.json")
with open(machine_config_json_path, "r") as local_config:
    local_config_data = json.load(local_config)

experiment_data_path = local_config_data.get("data_path")
experiment_data_folder = os.path.join(ROOT_DIR, experiment_data_path, "nuScenes_dataset_in_AtCity_format", "vjaiswalEnd2End", "data")
experiment_data_root = Path(experiment_data_folder)

#data_root = './TrainingTooling_Data/nuScenes_dataset_in_AtCity_format/vjaiswalEnd2End/data/'
folder_path = os.path.join(experiment_data_root, "val_keyframes")  # Replace with the actual folder path
correct_npy_files = []

# Iterate through the files in the folder
for idx, file_name in enumerate(os.listdir(folder_path)):
    if file_name.endswith(".npy"):
        file_path = os.path.join(folder_path, file_name)
        try:
            # Attempt to load the file to check for pickled data
            data =np.load(file_path, allow_pickle=False)
            correct_npy_files.append((file_name))
        except ValueError as e:
            if "Cannot load file containing pickled data" in str(e):
                print(f"Skipping file {file_name} due to pickled data issue")
                continue  # Skip to the next iteration
                #npy_files_with_pickled_issue.append((idx, file_name))

print("Number of correct .npy files:", len(correct_npy_files))
