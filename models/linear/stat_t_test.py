# performs t test on the brain data
"""
This script intends to perform t test on the coefficients from lasso regression
Author: Tiankang Xie & Team
"""
from nltools.data import Brain_Data, Adjacency,Design_Matrix
import glob
import nibabel as nib
import nilearn as nil
from nilearn.image import resample_img
import numpy as np
import pickle
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm import tqdm

all_subjects = glob.glob('/home/ubuntu/hcp_data/ten_subjects/100610/MNINonLinear/Results/*.nii.gz')
orig_nii = nib.load(all_subjects[0])
downsampled_nii = resample_img(orig_nii, target_affine=np.eye(3)*3., interpolation='nearest')

lasso_coefs_path = '/home/ubuntu/tk_trial/t_test/'
all_coef_data = glob.glob('/home/ubuntu/tk_trial/lasso*.pkl')
#WRITE_T_MAP_DIR = '/home/ubuntu/tk_trial/'+'v_all_subj_ind_t.nii.gz'
RESULT_DIR = '/home/ubuntu/tk_trial/LSTM/results/'
GROUP_RESULT_DIR = '/home/ubuntu/tk_trial/LSTM/results/cross_subject/'
NJOBS = 12
NCOMP = 81
# dump all coefficients into nilearn images
for i,coef_dat in enumerate(all_coef_data):
    with open(coef_dat,'rb') as fp:
        lasso_coeffs = pickle.load(fp)

    lasso_coeffs = np.concatenate(lasso_coeffs,0)
    brainA = lasso_coeffs.reshape(61,74,61,-1)
    for jj in range(brainA.shape[-1]):
        if not os.path.exists(os.path.join(RESULT_DIR,f"PC_{jj}")):
            os.mkdir(os.path.join(RESULT_DIR,f"PC_{jj}"))

        new_fmri = nil.image.new_img_like(downsampled_nii, brainA[...,jj])
        new_fmri.to_filename(os.path.join(RESULT_DIR, f"PC_{jj}", 'betaval'+str(i)+'.nii.gz'))

def _write_group_test(pcNum):
    try:
        RESULT_DIR = '/home/ubuntu/tk_trial/LSTM/results/'
        GROUP_RESULT_DIR = '/home/ubuntu/tk_trial/LSTM/results/cross_subject/'

        emo_list = glob.glob(RESULT_DIR+f'PC_{pcNum}/betaval*.nii.gz')
        # conduct statistical tests on the coefficients.
        emo_list.sort()
        emo_dat = Brain_Data(emo_list)
        emo_stats = emo_dat.ttest(threshold_dict={"fdr":.05})
        emo_stats["thr_t"].write(os.path.join(GROUP_RESULT_DIR,f'groupbeta_{pcNum}.nii.gz'))    
    except:
        return;

Parallel(n_jobs=NJOBS)(delayed(_write_group_test)(pcNum=jj) for jj in tqdm(range(NCOMP)))
