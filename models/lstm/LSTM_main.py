import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.autograd.variable import Variable
import gc
from tqdm import tqdm
from LSTMNet import fMRICNNLSTM, featureEncoder 
from LSTM_loader import fMRI_loader
from sklearn.model_selection import train_test_split

#===hyperparam=====
BATCH_SIZE = 2
N_EPOCH = 20
LR = 1e-3
TIME_LENGTH = 921
TRAIN_SIZE = 0.6
SAVE_PATH = '/home/ubuntu/tk_trial/LSTM/checkpoint/'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
VID_DIR = '/home/ubuntu/hcp_data/stimuli_jpg/'
fMRI_DIR = '/home/ubuntu/hcp_data/ten_subjects/'
MOVIE_NUM = '1' # use movie 1 as demo

video1 = f'7T_MOVIE1_CC{MOVIE_NUM}_v2_224x224_72_last_layer.npy'
fmri_subject_path = glob.glob(fMRI_DIR + '*/MNINonLinear/Results/tfMRI_MOVIE1_7T_*/tfMRI_MOVIE*_7T_*_hp2000_clean_resample.nii.gz')

# Loading data
data_loader = fMRI_loader(fmri_file_paths=fmri_subject_path,vid_feat_paths=VID_DIR + video1)

train_indices, split_indices = train_test_split(range(len(data_loader)), train_size=int(TRAIN_SIZE*len(data_loader)))
val_indices, test_indices = train_test_split(split_indices, train_size=int(0.5*len(split_indices)))

train_dat = torch.utils.data.dataset.Subset(data_loader,train_indices)
val_dat = torch.utils.data.dataset.Subset(data_loader,val_indices)
test_dat = torch.utils.data.dataset.Subset(data_loader,test_indices)

train_set = DataLoader(train_dat,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
val_set = DataLoader(val_dat,batch_size=BATCH_SIZE, shuffle=False,num_workers=0)
test_set = DataLoader(test_dat,batch_size=BATCH_SIZE, shuffle=False,num_workers=0)

del train_dat, val_dat, test_dat

# Loading models
model_fmri = fMRICNNLSTM(input_dim=1, intermediate_dim=64, hidden_dim=64, output_dim=40, input_time=TIME_LENGTH)
model_video = featureEncoder(input_dim=400,  hidden_dim=128, output_dim=40, time_resolution=24)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model_fmri = nn.DataParallel(model_fmri)
    model_video = nn.DataParallel(model_video)

model_fmri.to(device)
model_video.to(device)

# set up optimizer and loss function
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(params=list(model_fmri.parameters()) + list(model_video.parameters()),lr=LR, momentum=0.9, nesterov=True)
best_eval_score = 2**31-1

# Start training
for epoch_idx in range(N_EPOCH):

    loss_train = 0
    train_step = 0

    for batch_idx, (fmridat, viddat) in tqdm(enumerate(train_set)):
        optimizer.zero_grad()
        fmridat = Variable(fmridat).to(f'cuda:{model_fmri.device_ids[0]}').float()
        viddat = Variable(viddat).to(f'cuda:{model_video.device_ids[0]}', dtype=torch.float)
        fmri_out = model_fmri(fmridat)
        vid_out = model_video(viddat)
        loss = criterion(fmri_out.permute(0,2,1), vid_out)
        loss.backward()
        optimizer.step()
        loss_train += loss.detach().cpu().numpy()
        train_step += 1
    print(f'epoch: {epoch_idx}, train loss: {loss_train/train_step}')

    loss_val = 0
    val_step = 0
    for batch_idx, (fmridat, viddat) in enumerate(val_set):
        #optimizer.zero_grad()
        model_fmri.eval()
        model_video.eval()
        with torch.no_grad():
            fmridat = Variable(fmridat).to(f'cuda:{model_video.device_ids[0]}').float()
            viddat = Variable(viddat).to(f'cuda:{model_video.device_ids[0]}', dtype=torch.float)
            fmri_out = model_fmri(fmridat)
            vid_out = model_video(viddat)
            loss = criterion(fmri_out.permute(0,2,1), vid_out)
            loss_val += loss.detach().cpu().numpy()
            val_step += 1
    print(f'epoch: {epoch_idx}, validation loss: {loss_val/val_step}')
    
    if loss_val < best_eval_score:
        print(f'saving best run at epoch={epoch_idx}')
        torch.save(model_fmri.state_dict(), SAVE_PATH+f'model_fmri_{epoch_idx}')
        torch.save(model_video.state_dict(), SAVE_PATH+f'model_video_{epoch_idx}')

    

