import os
import sys
import json
import h5py
import numpy as np
from pathlib import Path
from natsort import natsorted


# Add the root folder to the Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.insert(0, ROOT_DIR)

machine_config_json_path = os.path.join("configs", "machine_configs", "machine_config_local.json")
with open(machine_config_json_path, "r") as local_config:
    local_config_data = json.load(local_config)
    logs_dir = local_config_data.get("model_path")
    logs_dir = os.path.join(ROOT_DIR, logs_dir)
    logs_dir = Path(logs_dir)
    model_logs_folder = logs_dir.joinpath('PointTransformerBaseline')
    model_logs_current_iteration = model_logs_folder.joinpath('12.1_training_with_moving_static_class_separated_1stTraining_ImprovedGTFurther_1.4_0.4_1.5_0.5')
    visualizer_predictions_dir = model_logs_current_iteration.joinpath('visualizer_predictions_100toLast')

    npy_predictions_dir = model_logs_current_iteration.joinpath('epoch40_npy_predictions_Sweeps_Included_142039')

val_h5_scene_files = natsorted(os.listdir(visualizer_predictions_dir))
predicted_npy_files = natsorted(os.listdir(npy_predictions_dir))

frames_covered = 0
for scene_file in val_h5_scene_files:
    print(scene_file)
    val_h5_scene_path = os.path.join(visualizer_predictions_dir, scene_file)

    #Load h5 file for corresponding scene
    with h5py.File(val_h5_scene_path, 'r+') as f:

        num_frames = (f['batch_to_radar_frame_map'].shape[0] - 1 )
        sensors_group = f['sensors']

        # prepare numpy array for each frame
        for i in range(num_frames):
            
            npy_frame_file = predicted_npy_files[frames_covered + i]
            npy_frame_file_path = os.path.join(npy_predictions_dir, npy_frame_file)
            npy_frame_data = np.load(npy_frame_file_path)
            npy_frame_data = np.squeeze(npy_frame_data, axis=0)
            predicted_label = np.argmax(npy_frame_data, axis=1)
            points_covered = 0
            for sensor_name in sensors_group.keys():
                sensor_group = sensors_group[sensor_name]

                if i <= (num_frames - 2):
                    slice_start = int(sensor_group['data_slices'][i])
                    slice_end = int(sensor_group['data_slices'][i+1])
                    if slice_end != slice_start:  
                        data_sensorwise = np.array(sensor_group['data'][slice_start:slice_end, :])
                        data_sensorwise[:, -1] = predicted_label[points_covered:(points_covered+data_sensorwise.shape[0])]
                        sensor_group['data'][slice_start:slice_end, :] = data_sensorwise
                        points_covered += data_sensorwise.shape[0]
                    else:
                        continue
                else:
                    slice_start = int(sensor_group['data_slices'][i])
                    data_sensorwise = np.array(sensor_group['data'][slice_start:, :])
                    data_sensorwise[:, -1] = predicted_label[points_covered:(points_covered+data_sensorwise.shape[0])]
                    sensor_group['data'][slice_start:, :] = data_sensorwise
                    points_covered += data_sensorwise.shape[0]
        f.close()
        frames_covered += num_frames
print(frames_covered)