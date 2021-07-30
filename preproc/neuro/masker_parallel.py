import os, nibabel as nib, nilearn
from glob import glob
from nilearn.input_data import NiftiMasker
from joblib import Parallel, delayed
import joblib
from tqdm import tqdm

all_subjects = glob('/home/ubuntu/hcp_data/ten_subjects/*/MNINonLinear/Results/tfMRI_MOVIE*_7T_*/tfMRI_MOVIE*_7T_*_hp2000_clean.nii.gz')

mask = '/home/ubuntu/tk_trial/hcp_MNImask_1.6mm.nii.gz' #'/home/ubuntu/tk_trial/hcp_MotionMask_1.6mm.nii.gz'

def MaskingFunc(subj_fn, mask_fn):
    masker = NiftiMasker(mask_img=mask_fn)
    mask = nib.load(mask_fn)

    file_dir = os.path.dirname(subj_fn)
    filename = os.path.basename(subj_fn)
    outname = os.path.join(file_dir, filename[:-7]+'_masked'+'.nii.gz')

    if not os.path.isfile(outname):
        print('running- ', outname)

        subj_data = nib.load(subj_fn)
        z = masker.fit_transform(subj_data)

        subj_data_masked = nilearn.masking.unmask(z, mask)
        nib.save(subj_data_masked, outname)

    else:
        print('skipping- ', outname)
        return

Parallel(n_jobs=12)(delayed(MaskingFunc)(subj_fn=a_subj, mask_fn=mask) for a_subj in tqdm(all_subjects))

# subj_arr_masked = subj_data_masked.get_fdata()

# plt.imshow(subj_arr_masked[:,:,60,100])
# plt.show()
 
