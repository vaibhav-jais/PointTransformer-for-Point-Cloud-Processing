import os
import argparse
import datetime
import json
import sys
import logging
import importlib
import shutil
import time
import distutils.version
import numpy as np
import torch
import warnings
from torch import nn as nn
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted

from torch.utils.tensorboard import SummaryWriter   # class to log data for visualization by Tensorboard
from build_dataset.dataset import class_weight
from build_dataset.dataloader import data_loader
from Loss_function.focal_loss import FocalLoss
from utils import SaveModelTorchscript

# Ignore the UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.insert(0, ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# check device to train on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device : {device}")
CUDA_LAUNCH_BLOCKING=1
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='PointNet2', help='model name [default: PointNet2]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=51, type=int, help='Epoch to run [default: 16]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--step_size', type=int, default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--momentum', default=0.9, type=float, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='Training_with_static_moving_separated_NOSWEEPS', help='Log path [default: None]')
    parser.add_argument('--weight_decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    NUM_CLASSES = 6
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epoch
    INITIAL_LEARNING_RATE = args.learning_rate
    LEARNING_RATE_DECCAY_RATE = args.lr_decay_rate
    STEP_SIZE = args.step_size
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM = args.momentum      # for SGD (if used)
    MOMENTUM_ORIGINAL = 0.5
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size
    #machine_config_local for local system
    machine_config_json_path = os.path.join("configs", "machine_configs", "machine_config_local.json")
    with open(machine_config_json_path, "r") as local_config:
        local_config_data = json.load(local_config)

    '''CREATE DIR'''
    experiment_logs_path = local_config_data.get("model_path")
    experiment_logs_folder = os.path.join(ROOT_DIR, experiment_logs_path)
    experiment_dir = Path(experiment_logs_folder)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('PointNet++_original_hyperparameters')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs_text/')
    log_dir.mkdir(exist_ok=True)
    tensorboard_logs_dir = experiment_dir.joinpath('tensorboard/')
    tensorboard_logs_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model") #creating a logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    #NUM_POINT = args.npoint
    
    print("start loading training data ...")
    experiment_data_path = local_config_data.get("data_path")
    experiment_data_folder = os.path.join(ROOT_DIR, experiment_data_path, "nuScenes_dataset_in_AtCity_format", "vjaiswalEnd2End", "data", "Data_with_stationary_moving_NOSWEEPS")
    experiment_data_root = Path(experiment_data_folder)
    trainDataLoader = data_loader(data_root=experiment_data_root, batch_size=BATCH_SIZE, train=True, num_workers=6, pin_memory=True)
    valDataLoader = data_loader(data_root=experiment_data_root, batch_size=1, train=False, num_workers=6, pin_memory=True)
    log_string('dataloader ran successfully ...')
    train_class_frequency = class_weight(experiment_data_root, train=True)
    #train_class_frequency= torch.tensor([255042., 1366084., 2008006., 115667., 36892., 29076158.], dtype=torch.float64)
    val_class_frequency = class_weight(experiment_data_root, train=False)
    #val_class_frequency= torch.tensor([62826., 258362., 374623., 21719., 8572., 6607891.], dtype=torch.float64)
    print(train_class_frequency)
    print(val_class_frequency)
    
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)    # import network module
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_modules.py', str(experiment_dir))
    model = MODEL.PointNet2Model(NUM_CLASSES).to(device)

    shutil.copy(sys.argv[0], str(experiment_dir))

    '''LOSS FUNCTION LOADING'''    
    loss_fn = FocalLoss()  # since the loss computation is a purely numerical operation and does not involve any model parameters. so no need to move to CUDA 
    
    def inplace_relu(m):
        classname = m.__class__.__name__
        if classname.find('ReLU' )!= -1:
            m.inplace = True
    model.apply(inplace_relu)

    def xavier_weight_initialization(m):
        classname = m.__class__.__name__
        if classname.find('Linear' )!= -1:
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Conv2d' )!= -1:
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Conv1d')!= -1:
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        
    model = model.apply(xavier_weight_initialization)

    start_epoch = 0
    try:
        checkpoint_epoch_load = torch.load(str(checkpoints_dir) + '/last_epoch.pth')
        start_epoch = checkpoint_epoch_load['epoch']
        model.load_state_dict(checkpoint_epoch_load['model_state_dict'])
        optimizer.load_state_dict(checkpoint_epoch_load['optimizer_state_dict'])
        log_string('Loading checkpoint from epoch %d' % (start_epoch))
    except:
        log_string('Starting training from beginning')

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=INITIAL_LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-08,
        weight_decay=args.weight_decay_rate
        )  
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LEARNING_RATE, momentum=MOMENTUM)

    def bn_momentum_adjust(m, MOMENTUM):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = MOMENTUM

    
    main_time_start = time.time()
    for epoch in range(start_epoch, NUM_EPOCHS):
        """Training the model given each frame as input"""
        log_string('**** Epoch (%d/%s) ****' % (epoch + 1, NUM_EPOCHS))
        lr = max(INITIAL_LEARNING_RATE * (LEARNING_RATE_DECCAY_RATE ** (epoch // STEP_SIZE)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        model.apply(lambda x:bn_momentum_adjust(x,momentum))

        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        total_loss = 0
        model = model.train()

        start_time = time.time()
        for batch_idx, (data_padded, labels_padded, mask, frame_name) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            "mask represents which elements are padded and with this we can remove the padded part in data and labels and use the original data and label"
            model = model.train()
            optimizer.zero_grad()
            data = data_padded*mask.unsqueeze(2)
            #data = data_padded
            data = data.float().to(device=device)
            labels_padded = labels_padded.long()
            labels = labels_padded.to(device=device)
            #labels = labels.long().to(device=device).T
            NUM_POINT = data.shape[1]
            predictions = model(data)
            predictions = predictions.contiguous().view(-1, NUM_CLASSES)
            labels = labels.reshape(-1, 1)[:, 0]
            batch_label = labels.reshape(-1, 1)[:, 0].cpu().data.numpy()
            train_loss = loss_fn(predictions, labels, train_class_frequency.to(device=device))
            #train_loss = loss_fn(labels, predictions, mask)
            if train_loss is None:
                log_string('Training loss was None at batch_num: %f' % (batch_idx+1))
                log_string('Corresponding frames: %s' % frame_name)
            #train_loss = loss_fn(predictions, labels.squeeze().long())
            train_loss.backward()
            optimizer.step()               

            pred_choice = predictions.cpu().data.max(1)[1].numpy()
            batch_correct = np.sum(pred_choice == batch_label)
            total_correct += batch_correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            total_loss += train_loss.item()
            #if batch_idx % 400 == 0:
                #log_string('train_loss , total_train_loss: %f/%f' % (train_loss, total_loss))
        mean_training_loss = total_loss / num_batches
        log_string('Training mean loss: %f' % (mean_training_loss))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)*100))
        
        writer = SummaryWriter(log_dir=tensorboard_logs_dir)
        writer.add_scalar('Loss/Training', mean_training_loss, epoch+1)

        
        #validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target, frame_name) in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
                data = data.float()
                target = target.long()
                prediction = model(data.to(device=device))
                prediction = prediction.contiguous().view(-1, NUM_CLASSES)
                target = target.reshape(-1, 1)[:, 0]
                val_loss = loss_fn(prediction, target.to(device=device), val_class_frequency.to(device=device))
                total_val_loss += val_loss.item()
        mean_val_loss = total_val_loss / len(valDataLoader)
        log_string('Validation mean loss: %f' % (mean_val_loss))
        writer.add_scalar('Loss/Validation', mean_val_loss, epoch+1)
        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = os.path.join(checkpoints_dir, f'epoch_{epoch}.pth')
            log_string('Saving checkpoint at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            # tracing the trained model and saving as TorchScript
            example_input_path = os.path.join(experiment_data_root, "train_keyframes")
            example_frames = natsorted(os.listdir(example_input_path))
            frame_file = example_frames[epoch]
            frame_file_path = os.path.join(example_input_path, frame_file)
            example_input = np.load(frame_file_path)
            example_input = example_input[np.newaxis, :]
            example_input = torch.tensor(example_input[:, :, [0, 1, 2, 7, 8, 15, 16]])
            example_input = example_input.to(device=device)
            torchscript_dir = experiment_dir.joinpath('torchscript/')
            torchscript_dir.mkdir(exist_ok=True)
            SaveModelTorchscript(model, example_input.float(), str(args.model), torchscript_dir, epoch)
            log_string('Saving TorchScript model....')
        end_time = time.time()
        log_string('******************** Epoch finished in (%d) minutes********************' % ((end_time - start_time)/60))
    writer.close()
    main_time_end = time.time()
    log_string('********************Training finished in (%d) hours********************' % ((main_time_end - main_time_start)/3600))

if __name__ == '__main__':
    args = parse_args()
    main(args)
