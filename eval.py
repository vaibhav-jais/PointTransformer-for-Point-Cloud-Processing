import os
import sys 
import torch
import torch.nn.functional as F
import numpy as np
from build_dataset.dataloader import data_loader
from tqdm import tqdm
from pathlib import Path
from models.PointNet2 import PointNet2Model

# Add the root folder to the Python path
root_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_folder)

# check device to train on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device : {device}")


def main():

    """Load the validation set and the traced model, generate prediction for each sample of a scene from validation data and
            save the prediction as numpy array (.npy format)

            val_loader (object): loads validation data of batch size 1 i.e one frame, corresponding labels and the name of the frame as string
            traced_model (.pt file): traced model that contains 
                            (a) the model's computataion graph
                            (b) captures the learned parameters(weights and biases) of the original model
                            (c) traced model converts operations and modules in the model into TorchScript function, which are more
                                                    optimized for execution
    """

    root = './data'
    val_loader = data_loader(data_root=root, batch_size=1, train=False, num_workers=2, pin_memory=0)
    num_classes = 6
    
    #Loading the serialized TorchScript representation of the model
    traced_model_dir = Path('./logs/')
    traced_model_dir = traced_model_dir.joinpath('PointNet++_25_epochs_batchsize_32')
    traced_model_dir = traced_model_dir.joinpath('2023-08-10_18-39')
    traced_model_dir = traced_model_dir.joinpath('torchscript')
    traced_model_dir = traced_model_dir.joinpath('PointNet2_epoch_49.pt')
    traced_model = torch.jit.load(traced_model_dir)
    
    #Loading checkpoints
    checkpoint_dir = Path('./logs/')
    checkpoint_dir = checkpoint_dir.joinpath('PointNet++_50_epochs_batchsize_32_newfocalloss')
    checkpoint_dir = checkpoint_dir.joinpath('2023-08-11_16-08')
    checkpoint_dir = checkpoint_dir.joinpath('checkpoints')
    checkpoint_dir = checkpoint_dir.joinpath('epoch_48.pth')
    checkpoints_dict = torch.load(checkpoint_dir)

    
    # path to store the model's prediction during inference
    predictions_logs_dir = Path('./logs/')
    predictions_logs_dir = predictions_logs_dir.joinpath('PointNet++_50_epochs_batchsize_32_newfocalloss')
    predictions_logs_dir = predictions_logs_dir.joinpath('2023-08-11_16-08')
    predictions_logs_checkpoint_npy = predictions_logs_dir.joinpath('checkpoint_epoch48_npy_predictions')
    predictions_logs_checkpoint_npy.mkdir(exist_ok=True)


    model = PointNet2Model(num_classes)
    model.load_state_dict(checkpoints_dict['model_state_dict'])
    print(checkpoints_dict['epoch'])
    for data, _ , frame_name in tqdm(val_loader):       # _ -> labels
        data = data.to(device=device)
        prediction_file_name = frame_name[0].split('.npy')[0]
        #traced_model.eval()
        model.eval()
        with torch.no_grad():
            model = model.to(device=device)
            predictions = F.softmax(model(data.float()), dim=-1)
            #predictions = torch.argmax(predictions,  dim=1, keepdim=True)
            predictions_save = predictions.detach().cpu().numpy()
            npy_filename = os.path.join(predictions_logs_checkpoint_npy, f'{prediction_file_name}.npy')
            np.save(npy_filename, predictions_save)

    return 

if __name__ == "__main__":
    main()

