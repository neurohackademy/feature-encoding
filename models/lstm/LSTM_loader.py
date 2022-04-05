"""
This script intends to Load and prepare data for deep learning.
Want to load the fMRI data and load the 
Author: Tiankang Xie & Team
"""
import pandas as pd
import nibabel
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from nltools.data import Brain_Data, Design_Matrix
import torch
import numpy as np
from torch.nn.functional import normalize
import torch.nn.functional as F
import nibabel as nib
import nilearn as nil
import glob

# fMRI_DIR = '/home/ubuntu/hcp_data/ten_subjects/'

# all_subjects = glob.glob(fMRI_DIR + '*/MNINonLinear/Results/tfMRI_MOVIE*_7T_*/tfMRI_MOVIE*_7T_*_hp2000_clean_resample.nii.gz')

# fda = nib.load(all_subjects[0])
# fda1 = fda.get_fdata()
# print('true')
# Classification by voxel does not work well. Let's start from classification from ROIs
# x: 6-55 -> 49
# y: 7-67 -> 60
# z: 0:51 -> 51

class fMRI_loader(Dataset):

    def __init__(self, fmri_file_paths, vid_feat_paths):
        # Some predefined path 
        #self.gen_dir = gen_dir
        self.fmri_file_paths = fmri_file_paths
        self.vid_feat_paths = vid_feat_paths

    def __getitem__(self, index): 

        curr_subject = self.fmri_file_paths[index]
        # Load in 16 different emotions
        brain = nibabel.load(curr_subject)
        bdata = brain.get_fdata()
        bdata = np.nan_to_num(bdata)
        bdata = bdata[6:55,7:67,0:51,:]
        # of shape 47, 60, 47
        bdata = torch.from_numpy(np.expand_dims(bdata,0))
        p2d = (0,0,4,5,0,0,5,6) # pad to same shape (cube)
        bdata = F.pad(bdata, p2d, "constant", 0)
        bdata = bdata.permute(4,0,1,2,3)
        vid_feat = torch.from_numpy(np.load(self.vid_feat_paths, allow_pickle=True))
        vid_feat = vid_feat.permute(1,0)
        return bdata, vid_feat
    
    def __len__(self):
        return len(self.fmri_file_paths)