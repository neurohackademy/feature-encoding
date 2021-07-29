"""
Linear Lasso-PCR, outputs coefficient
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

# Read in video features
VID_DIR = '/home/ubuntu/hcp_data/stimuli_jpg/'
fMRI_DIR = '/home/ubuntu/hcp_data/ten_subjects/'
WRITE_PATH = '/home/ubuntu/tk_trial/' # where to write the lasso coefficient files
NJOBS = 4 # parallel processing of the lasso coefs


video1 = '7T_MOVIE1_CC1_v2_224x224_72_last_layer.npy'
video2 = '7T_MOVIE2_HO1_v2_224x224_72_last_layer.npy'
video3 = '7T_MOVIE3_CC2_v2_224x224_72_last_layer.npy'
video4 = '7T_MOVIE4_HO2_v2_224x224_72_last_layer.npy'

vid1_feat = np.load(VID_DIR + video1, allow_pickle=True)
vid2_feat = np.load(VID_DIR + video2, allow_pickle=True)
vid3_feat = np.load(VID_DIR + video3, allow_pickle=True)
vid4_feat = np.load(VID_DIR + video4, allow_pickle=True)

# plt.plot(vid1_feat.T[0,:],color='red')
# plt.plot(np.arange(0,22103,24), tensor_rolled_data[0,:],color='blue')
# plt.savefig("/home/ubuntu/tk_trial/sanity.jpg")

#=====================preprocess x data
# from nilearn.masking import compute_gray_matter_mask, apply_mask

# for i in range(vid1_feat.shape[1]):
#     plt.plot(vid1_feat[:,i])
# plt.savefig("/home/ubuntu/tk_trial/try1.jpg")
# fmridat = nib.load(all_subjects[0])
# #fmridd = np.expand_dims(fmridat.get_fdata(),0)
# print('get data')

#TODO
# preprocess and do resampling data
# def _preprocess(filenames, writepath):
#     """
#     preprocess data, resampling, apply grey matter mask, write to local dir
#     """
#     return;

#mask_img = nib.load('/home/ubuntu/tk_trial/MNI152NLin2009cAsym_desc-thr25gray1mm_mask.nii')
# = compute_gray_matter_mask(func_filename)
#masked_data = apply_mask(func_filename, mask_img)

# /home/ubuntu/hcp_data/ten_subjects/109123/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/tfMRI_MOVIE1_7T_AP_hp2000_clean.nii.gz

#make the x data (fmri for each movie)
#need to 1. mask gray matter out, 2. downsample voxel size to 3/4 mm

# fmri_cbd_dat = (fmri_cbd_dat - np.mean(fmri_cbd_dat,axis=1,keepdims=True))/np.std(fmri_cbd_dat,axis=1,keepdims=True)
# VxT

print('get data')

def _prepare_xdat(movie_num):
    all_subjects = glob.glob(fMRI_DIR + f'*/MNINonLinear/Results/tfMRI_MOVIE{movie_num}_7T_*/tfMRI_MOVIE{movie_num}_7T_*_hp2000_clean.nii.gz')

    subj_data = None
    for subj in all_subjects:
        fmridat = nib.load(subj)
        fmridd = np.expand_dims(fmridat.get_fdata(),0)

        if subj_data is None:
            subj_data = fmridd
        else:
            subj_data = np.concatenate([subj_data, fmridd],axis=0)
    
    return subj_data

#===================================================================================
def _prepare_ydat(ydat, win_size):
    """
    prepare y data labels by using a sliding window and averaging
    Input: 
        ydat: np array of shape (Time, F)
        window_length: sliding window/moving average length
    """
    tensor_rolled_data = np.mean(ydat.reshape(ydat.shape[0], -1, win_size),axis=2)
    return tensor_rolled_data

ydat_downsampled1 = _prepare_ydat(vid1_feat.T, win_size=24)
ydat_downsampled2 = _prepare_ydat(vid2_feat.T, win_size=24)
ydat_downsampled3 = _prepare_ydat(vid3_feat.T, win_size=24)
ydat_downsampled4 = _prepare_ydat(vid4_feat.T, win_size=24)
ydat_downsample = np.concatenate([ydat_downsampled1,ydat_downsampled2,ydat_downsampled3,ydat_downsampled4],1)
ydat_downsample = (ydat_downsample - np.mean(ydat_downsample,axis=0))/np.std(ydat_downsample,axis=0)
PCA_model = PCA(0.95)
vid_pca = PCA_model.fit_transform(ydat_downsample.T)

# pca = PCA().fit(ydat_downsample.T)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.savefig("/home/ubuntu/tk_trial/pca_explained.jpg")

# x: PCAxT
# y: VxT
# Configurate and run linear model

ALPHA=0.1 # lasso coefficient
all_subjects = glob.glob(fMRI_DIR + '*/MNINonLinear/Results/tfMRI_MOVIE*_7T_*/tfMRI_MOVIE*_7T_*_hp2000_clean_resample.nii.gz')
all_subj_names = [i.split('/')[5] for i in all_subjects]
all_subj_names = np.unique(all_subj_names)

def run_lasso_regrs(X,Y):
    if np.all(Y==0):
        return np.zeros((1,X.shape[1]))
    else:
        clf = Lasso(alpha=ALPHA)
        clf.fit(X,Y)
        return (clf.coef_.reshape(1,-1))


for i, ind_name in enumerate(all_subj_names):

    if os.path.exists(WRITE_PATH + f'lasso_coefs-{ind_name}.pkl'):
        continue;
        
    tmp_subj_dir = [m for m in all_subjects if ind_name in m]

    fmri_cbd_dat = None
    for subjj in tmp_subj_dir:
        fmridat = nib.load(subjj)
        fmridd = fmridat.get_fdata()
        if fmri_cbd_dat is None:
            fmri_cbd_dat = fmridd
        else:
            fmri_cbd_dat = np.concatenate([fmri_cbd_dat,fmridd], axis=-1)

    del fmridat, fmridd

    fmri_cbd_dat = np.swapaxes(fmri_cbd_dat.reshape(-1,fmri_cbd_dat.shape[-1]),0,1)
    
    LASSO_coefficient = np.zeros((vid_pca.shape[1],fmri_cbd_dat.shape[1]))

    lasso_coeffs = Parallel(n_jobs=NJOBS)(delayed(run_lasso_regrs)(X=vid_pca, Y=fmri_cbd_dat[:,i]) for i in tqdm(range(fmri_cbd_dat.shape[1])))

    import pickle
    with open(WRITE_PATH + f'lasso_coefs-{ind_name}.pkl','wb') as fp:
        pickle.dump(lasso_coeffs, fp)
