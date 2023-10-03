import json
import sys
import os
import h5py
import numpy as np
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

machine_config_json_path = os.path.join(ROOT_DIR, "configs", "machine_configs", "machine_config_local.json")
with open(machine_config_json_path, "r") as local_config:
    local_config_data = json.load(local_config)

experiment_data_path = local_config_data.get("data_path")
experiment_data_folder = os.path.join(ROOT_DIR, experiment_data_path, "nuScenes_dataset_in_AtCity_format", "vjaiswalEnd2End", "data", "Data_with_stationary_moving_NOSWEEPS")
experiment_data_root = Path(experiment_data_folder)
train_keyframes_path = os.path.join(experiment_data_root, "train_keyframes")
val_keyframes_path = os.path.join(experiment_data_root, "val_keyframes")

h5_data_path = local_config_data.get("h5_path")
h5_data_root = Path(h5_data_path)
train_h5scenes_path =os.path.join(h5_data_root, 'train')
val_h5scenes_path =os.path.join(h5_data_root, 'val')


scene_files = sorted(os.listdir(val_h5scenes_path))
for scene_file in scene_files:
    print(scene_file)
    scene_path = os.path.join(val_h5scenes_path, scene_file)

    #Load h5 file for corresponding scene
    with h5py.File(scene_path, 'r') as f:

        num_frames = (f['batch_to_radar_frame_map'].shape[0] - 1 )
        sensors_group = f['sensors']

        # prepare numpy array for each frame
        for i in range(num_frames):
            data_array = np.empty((0,18))
            for sensor_name in sensors_group.keys():
                sensor_group = sensors_group[sensor_name]

                if i <= (num_frames - 2):
                    slice_start = int(sensor_group['data_slices'][i])
                    slice_end = int(sensor_group['data_slices'][i+1])
                    if slice_end != slice_start:
                        data_sensorwise = np.array(sensor_group['data'][slice_start:slice_end, :])
                        data_array = np.vstack((data_array, data_sensorwise))
                    else:
                        continue
                else:
                    slice_start = int(sensor_group['data_slices'][i])
                    data_sensorwise = np.array(sensor_group['data'][slice_start:, :])
                    data_array = np.vstack((data_array, data_sensorwise))

            file_name = scene_file.split('.h5')[0]
            npy_filename = os.path.join(val_keyframes_path, f'{file_name}_frame{i+1}.npy')
            np.save(npy_filename, data_array)
        f.close()
