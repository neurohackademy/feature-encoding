# performs t test on the brain data
"""
This script intends to plot the brain plots.
Author: Tiankang Xie & Team
"""
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import torch
import matplotlib.pyplot as plt
import numpy as np
# read in fmri data
import os
import glob
import nibabel as nib
import nilearn as nil
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
from nilearn.image import resample_img
import nibabel as nb
from nltools.data import Brain_Data, Adjacency,Design_Matrix

# plot the t test map:
from nilearn import plotting
# Whole brain sagittal cuts and map is thresholded at 3
#all_t_obj = nb.load(os.path.join('/home/ubuntu/tk_trial/','v_all_subj_ind_t.nii.gz')) 
#plotting.plot_glass_brain(all_t_obj)
#plt.imshow(brainA[:,:,30,0])
#plt.colorbar()
PATH_DIR = '/home/ubuntu/tk_trial/LSTM/results/cross_subject/'
COMP_NUM = 90
for i in range(COMP_NUM):
    try:
        PATH_TO_TMAP = os.path.join(PATH_DIR,f'groupbeta_{i}.nii.gz') # where to find the data
        WRITE_IMG_PATH = f'/home/ubuntu/tk_trial/brain_t_plot_COMP{i}.jpg' # where to store the global t statistics map

        emo_stats = Brain_Data(PATH_TO_TMAP)
        emo_stats.plot()
        plt.savefig(WRITE_IMG_PATH)
    except:
        continue;