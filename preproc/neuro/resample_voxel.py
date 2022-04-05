"""
This script intends to downsample HCP data into 3mm voxel size
Author: Tiankang Xie & Team
"""
import os
import glob
from nilearn.image import resample_img
import pylab as plt
import nibabel as nb
import numpy as np
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed

NUM_CORES = 2
all_subjects_paths = glob.glob('/home/ubuntu/hcp_data/ten_subjects/*/MNINonLinear/Results/tfMRI_MOVIE*_7T_*/tfMRI_MOVIE*_7T_*_hp2000_clean.nii.gz')

def _run_downsampling(subjj):
    if os.path.exists(os.path.join(os.path.dirname(subjj),os.path.basename(subjj).split('.')[0]+'_resample'+os.path.basename(subjj)[-7::])):
        print('file exists, skipping')
        return;
    else:
        orig_nii = nb.load(subjj)
        downsampled_nii = resample_img(orig_nii, target_affine=np.eye(3)*3., interpolation='nearest')
        noise_fmri = nil.image.new_img_like(downsampled_nii, brainA)
        downsampled_nii.to_filename(os.path.join(os.path.dirname(subjj),os.path.basename(subjj).split('.')[0]+'_resample'+os.path.basename(subjj)[-7::]))

Parallel(n_jobs=NUM_CORES)(delayed(_run_downsampling)(subjj=a_subj) for a_subj in tqdm(all_subjects_paths))
