import h5py
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import json
from build_dataset.dataloader import data_loader
from models.PointNet2 import PointNet2Model
from models.PointTransformerNetwork import PointTransformerSeg
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from natsort import natsorted
from pathlib import Path


# Add the root folder to the Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.insert(0, ROOT_DIR)

# check device to train on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device : {device}")

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='PointTransformerNetwork', help='model name [default: PointTransformerNetwork]')
    #parser.add_argument('--model', type=str, default='PointNet++', help='model name [default: PointTransformerNetwork]')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during validation [default: 16]')
    parser.add_argument('--nneighbors', type=int, default=16, help='nearser neighbors in KNN [default: 16]')
    parser.add_argument('--num_points', default=1024, type=int, help='num points as input to point transformer [default: 512]')
    parser.add_argument('--model_logs_folder', type=str, default='PointTransformerBaseline', help='Models Log path [default: None]')
    #parser.add_argument('--model_logs_folder', type=str, default='PointNet++_original_hyperparameters', help='Models Log path [default: None]')
    parser.add_argument('--model_logs_current_iteration', type=str, default='Training_with_relativetimestamp', help='Models Log subpath [default: None]')
    parser.add_argument('--best_model', type=str, default='epoch_30.pth', help='best model parameter to use for validation [default: epoch_25.pth]')
    parser.add_argument('--save_npy_predictions', type=str, default='epoch30_npy_predictions_Sweeps_Included', help='folder name to save npy preds [default: None]')

    return parser.parse_args()

def generate_confusion_matrix():
    """ Generates a single confusion matrix of shape 6*6 by combining combining confusion matrix of each frame and for all the scenes

    Args:
        val_h5_folder_path (str): folder path that contains 150 h5 scenes for validation set
    """
    args = parse_args()
    machine_config_json_path = os.path.join("configs", "machine_configs", "machine_config_local.json")
    with open(machine_config_json_path, "r") as local_config:
        local_config_data = json.load(local_config)
    
    validation_data_path = local_config_data.get("data_path")
    validation_data_folder = os.path.join(ROOT_DIR, validation_data_path, "nuScenes_dataset_in_AtCity_format", "vjaiswalEnd2End", "data")
    validation_data_root = Path(validation_data_folder)
    val_loader = data_loader(data_root=validation_data_root, batch_size=1, train=False, num_workers=6, pin_memory=0)
    val_h5_folder_path = os.path.join(validation_data_root, "val_h5", "")
    
    logs_dir = local_config_data.get("model_path")
    logs_dir = os.path.join(ROOT_DIR, logs_dir)
    logs_dir = Path(logs_dir)

    model_logs_folder = logs_dir.joinpath(args.model_logs_folder)
    model_logs_current_iteration = model_logs_folder.joinpath(args.model_logs_current_iteration)
    """
    #Loading the serialized TorchScript representation of the model
    traced_model_dir = model_logs_current_iteration.joinpath('torchscript')
    traced_model_dir = traced_model_dir.joinpath('PointNet2_epoch_25.pt')
    traced_model = torch.jit.load(traced_model_dir)
    predictions_logs_torchscript_npy = model_logs_current_iteration.joinpath('torchscript_epoch25_npy_predictions')
    predictions_logs_torchscript_npy.mkdir(exist_ok=True)
    """
    
    #Loading checkpoints
    checkpoint_dir = model_logs_current_iteration.joinpath('checkpoints')
    checkpoint_dir = checkpoint_dir.joinpath(args.best_model)
    checkpoints_dict = torch.load(checkpoint_dir)
    
    # path to store the model's prediction during inference
    predictions_logs_checkpoint_npy = model_logs_current_iteration.joinpath(args.save_npy_predictions)
    predictions_logs_checkpoint_npy.mkdir(exist_ok=True)
    
    num_classes = 6
    
    if args.model == 'PointTransformerNetwork':
        model = PointTransformerSeg(args.num_points, args.nneighbors)
    else:
        model = PointNet2Model(num_classes)
    
    model.load_state_dict(checkpoints_dict['model_state_dict'])
    print(checkpoints_dict['epoch'])
    
    for data, _ , frame_name in tqdm(val_loader):       # _ -> labels
        data = data.to(device=device)
        #data = data.squeeze()
        prediction_file_name = frame_name[0].split('.npy')[0]
        #traced_model.eval()
        model.eval()
        with torch.no_grad():
            model = model.to(device=device)
            #traced_model = traced_model.to(device=device)
            predictions = F.softmax(model(data.float()), dim=-1)
            #predictions = F.softmax(traced_model(data.float()), dim=-1)
            #predictions = torch.argmax(predictions,  dim=1, keepdim=True)
            predictions_save = predictions.detach().cpu().numpy()
            npy_filename = os.path.join(predictions_logs_checkpoint_npy, f'{prediction_file_name}.npy')
            #npy_filename = os.path.join(predictions_logs_torchscript_npy, f'{prediction_file_name}.npy')
            np.save(npy_filename, predictions_save)
    
    predicted_results_framewise = natsorted(os.listdir(predictions_logs_checkpoint_npy))
    #predicted_results_framewise = natsorted(os.listdir(predictions_logs_torchscript_npy))
    val_h5_scenes = natsorted(os.listdir(val_h5_folder_path))
    idx_scene = 0
    total_conf_matrix = np.zeros((6,6), dtype=np.int32)
    for val_scene in val_h5_scenes:
        val_file_path = os.path.join(val_h5_folder_path, val_scene)

        with h5py.File(val_file_path, 'r') as val_file:
            num_frames = (val_file['batch_to_radar_frame_map'].shape[0] - 1 )
            sensors_group = val_file['sensors'] 
            for frame_idx in range(num_frames):
                true_label_frame = np.empty((0, ))
                for sensor_name in sensors_group.keys():
                    sensor_group = sensors_group[sensor_name]
                    if frame_idx <= (num_frames - 2):
                        slice_start = int(sensor_group['data_slices'][frame_idx])
                        slice_end = int(sensor_group['data_slices'][frame_idx+1])
                        if slice_end != slice_start:
                            data_label_sensorwise = np.array(sensor_group['data'][slice_start:slice_end, -1])
                            true_label_frame = np.concatenate((true_label_frame, data_label_sensorwise), axis = 0)   
                        else:
                            pass
                    else:
                        slice_start = int(sensor_group['data_slices'][frame_idx])
                        data_label_sensorwise = np.array(sensor_group['data'][slice_start:, -1])
                        true_label_frame = np.concatenate((true_label_frame, data_label_sensorwise), axis = 0)

                predicted_results_frame = os.path.join(predictions_logs_checkpoint_npy , predicted_results_framewise[idx_scene + frame_idx])
                #predicted_results_frame = os.path.join(predictions_logs_torchscript_npy , predicted_results_framewise[idx_scene + frame_idx])
                predicted_label_frame = np.load(predicted_results_frame)
                predicted_label_frame = np.reshape(predicted_label_frame, (predicted_label_frame.shape[1], predicted_label_frame.shape[2]))
                predicted_label = np.argmax(predicted_label_frame, axis=1)

                assert true_label_frame.shape == predicted_label.shape, f"label_array of shape {true_label_frame.shape}, predicted_frame_label of shape {predicted_label_frame.shape}"

                conf_matrix_frame = confusion_matrix(true_label_frame, predicted_label, labels=list(range(num_classes)))
                total_conf_matrix[:conf_matrix_frame.shape[0], :conf_matrix_frame.shape[1]] += conf_matrix_frame
            idx_scene = idx_scene + num_frames
        val_file.close()
    
    return  total_conf_matrix, model_logs_current_iteration


def calculate_f1_score(final_confusion_matrix):
    
    precision = np.diag(final_confusion_matrix) / np.sum(final_confusion_matrix, axis=0)  # sum along the rows (vertically)
    recall = np.diag(final_confusion_matrix) / np.sum(final_confusion_matrix, axis=1)     # sum along the columns (horizontally)
    f1_score = 2 * (precision * recall) / (precision + recall)
    # Calculate macro-average F1 score
    macro_avg_f1 = np.mean(f1_score)
    print("Precision for each class:", precision)
    print("Recall for each class:", recall)
    print("F1 Score for each class:", f1_score)
    print("Macro-Average F1 Score:", macro_avg_f1)


def plot_conf_heatmap():

    final_confusion_matrix, model_logs_current_iteration = generate_confusion_matrix()
    calculate_f1_score(final_confusion_matrix)
    # Normalize the confusion matrix to get percentage proportions
    sum_per_row = np.sum(final_confusion_matrix, axis=1, keepdims=True)
    confusion_matrix_percentage = final_confusion_matrix / sum_per_row
    # Create subplots for the absolute and percentage heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    sn.set(font_scale=1.4) # for label size
    categories = ['Other', 'Vehicle_Moving', 'Vehicle_Stationary', 'Pedestrian', 'Bike', 'Background']

    # Plot the absolute confusion matrix heatmap
    ax1 = sn.heatmap(final_confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix (Absolute)')
    # set x-axis label and ticks. 
    ax1.set_xlabel("Predicted Classes", fontsize=14, labelpad=20)
    ax1.xaxis.set_ticklabels(categories)
    # set y-axis label and ticks
    ax1.set_ylabel("Actual Classes", fontsize=14, labelpad=20)
    ax1.yaxis.set_ticklabels(categories)

    # Plot the percentage confusion matrix heatmap
    ax2 = sn.heatmap(confusion_matrix_percentage, annot=True, fmt=".2%", cmap='Blues', ax=axes[1])
    axes[1].set_title('Confusion Matrix (Percentage)')
    # set x-axis label and ticks. 
    ax2.set_xlabel("Predicted Classes", fontsize=14, labelpad=20)
    ax2.xaxis.set_ticklabels(categories)
    # set y-axis label and ticks
    ax2.set_ylabel("Actual Classes", fontsize=14, labelpad=20)
    ax2.yaxis.set_ticklabels(categories)
    plt.tight_layout()
    plt.tight_layout()
    conf_file_name = os.path.join(model_logs_current_iteration, 'confusion_matrix_Sweeps_Included.png')
    plt.savefig(conf_file_name)
    plt.show()

    return final_confusion_matrix

if __name__ == '__main__':
    plot_conf_heatmap()
