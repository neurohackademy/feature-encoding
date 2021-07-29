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

all_subjects = glob.glob('/home/ubuntu/hcp_data/ten_subjects/100610/MNINonLinear/Results/*.nii.gz')
orig_nii = nib.load(all_subjects[0])
downsampled_nii = resample_img(orig_nii, target_affine=np.eye(3)*3., interpolation='nearest')

lasso_coefs_path = '/home/ubuntu/tk_trial/t_test/'
all_coef_data = glob.glob('/home/ubuntu/tk_trial/lasso*.pkl')
WRITE_T_MAP_DIR = '/home/ubuntu/tk_trial/'+'v_all_subj_ind_t.nii.gz'

# dump all coefficients into nilearn images
for i,coef_dat in enumerate(all_coef_data):
    with open(coef_dat,'rb') as fp:
        lasso_coeffs = pickle.load(fp)

    lasso_coeffs = np.concatenate(lasso_coeffs,0)
    brainA = lasso_coeffs.reshape(61,74,61,-1)
    noise_fmri = nil.image.new_img_like(downsampled_nii, brainA)
    noise_fmri.to_filename(os.path.join(lasso_coefs_path, 'betaval'+str(i)+'.nii.gz'))

# conduct statistical tests on the coefficients.
emo_list = glob.glob(os.path.join(lasso_coefs_path, 'beta*.nii.gz'))
emo_list.sort()
emo_dat = Brain_Data(emo_list)
emo_stats = emo_dat.ttest(threshold_dict={"fdr":.05})
emo_stats["thr_t"].write(os.path.join(WRITE_T_MAP_DIR))    
