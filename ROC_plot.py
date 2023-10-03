import os
import sys
import numpy as np
from pathlib import Path
import h5py
import json
from natsort import natsorted
import matplotlib.pyplot as plt

# Add the root folder to the Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.insert(0, ROOT_DIR)

def get_tpr_fnr_fpr_tnr(cm):
    """
    This function returns class-wise TPR, FNR, FPR & TNR
    [[cm]]: a 2-D array of a multiclass confusion matrix
            where horizontal axes represent actual classes
            and vertical axes represent predicted classes
    {output}: a dictionary of class-wise accuracy parameters
    """
    dict_metric = dict()
    num_classes = len(cm[0])
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    array_sum = sum(sum(cm))
    #initialize a blank nested dictionary
    for i in range(1, num_classes+1):
        keys = str(i)
        dict_metric[keys] = {"TPR":0, "FNR":0, "FPR":0, "TNR":0}
    # calculate and store class-wise TPR, FNR, FPR, TNR
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                keys = str(i+1)
                tp = cm[i, j]
                fn = row_sums[i] - cm[i, j]
                dict_metric[keys]["TPR"] = tp / (tp + fn)
                dict_metric[keys]["FNR"] = fn / (tp + fn)
                fp = col_sums[i] - cm[i, j]
                tn = array_sum - tp - fn - fp
                dict_metric[keys]["FPR"] = fp / (fp + tn)
                dict_metric[keys]["TNR"] = tn / (fp + tn)
    return dict_metric


def main():

    machine_config_json_path = os.path.join("configs", "machine_configs", "machine_config_local.json")
    with open(machine_config_json_path, "r") as local_config:
        local_config_data = json.load(local_config)

    logs_dir = local_config_data.get("model_path")
    logs_dir = os.path.join(ROOT_DIR, logs_dir)
    logs_dir = Path(logs_dir)
    conf_matrix_dir = logs_dir.joinpath('PointTransformerBaseline')
    conf_matrix_dir = conf_matrix_dir.joinpath('sampling_rate_4')
    conf_matrix_dir = conf_matrix_dir.joinpath('torchscript_epoch9_h5_confusion_matrix')
    conf_matrices = natsorted(os.listdir(conf_matrix_dir))
    score_thresholds = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52,
                    0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98]
    
    sum_conf_matrices = np.zeros((45,6,6), dtype=np.int32)
    for matrix in conf_matrices:
        each_matrix_path = os.path.join(conf_matrix_dir, matrix)
        with h5py.File(each_matrix_path, 'r') as matrix_file:
            nocare_false = matrix_file['nocare_false']   
            matrix_data = nocare_false['confus']  # order is true, pred, num_thresholds
            #creating an empty confusion matrix of shape 6*6 initialized with zeros
            matrix_data = np.array(matrix_data).transpose(2, 0, 1)           # order changed to num_thresholds, true, pred
            sum_conf_matrices += matrix_data

    class_colors = ['red', 'blue', 'orange', 'green', 'yellow', 'black']
    line_styles = ['-', '--', '-.', ':', 'solid', 'dashed']
    Classes = {'0': 'Background','1': 'Vehile Large','2': 'Vehicle','3': 'Pedestrian','4': 'Bike','5': 'Other'}

    plt.figure()
    for class_idx in range(6):
        tpr_list = []
        fpr_list = []
        for i in range(len(score_thresholds)):
            dict_metric = get_tpr_fnr_fpr_tnr(sum_conf_matrices[i, :, :])   
            fpr = dict_metric[str(class_idx + 1)]['FPR']
            fpr_list.append(fpr)
            tpr = dict_metric[str(class_idx + 1)]['TPR']
            tpr_list.append(tpr)

        plt.plot(fpr_list, tpr_list, label=f'Class {Classes[str(class_idx)]}', color=class_colors[class_idx], linestyle=line_styles[class_idx])

    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.show()

main()
